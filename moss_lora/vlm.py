"""OpenAI-compatible VLM client for llama.cpp caption/layout annotation.

The client is deliberately transport-only: it never loads model weights and
never downloads anything. Point it at a local ``llama-server`` (or any
OpenAI-compatible ``/v1/chat/completions`` endpoint) and it sends one base64
image plus the versioned prompt contract, validates the response with
``moss_lora.annotations`` and retries up to three times:

1. combined caption + layout call at the configured temperature;
2. combined retry at low temperature with validator errors appended;
3. split caption call then layout call (OCR boxes passed as fixed context).

A second (Qwen) configuration can be supplied as a fallback; the fallback is
called only after the primary model exhausts its attempts. Nothing here claims
anything about caption/bbox quality -- the bakeoff is a separate gate.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageOps

from moss_lora.annotations import (
    BOX_FORMAT,
    NORMALIZED_SPACE,
    AnnotationError,
    Layout,
    VLMAnnotation,
    parse_layout,
    parse_raw_caption,
    parse_vlm_annotation,
    strip_metadata_labels,
)

PROMPT_VERSION = "moss-caption-v4"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "docs" / "caption_prompt.txt"
MAX_ATTEMPTS = 3
REFUSAL_RE = re.compile(
    r"(?i)\b(i can(?:no|')t|i cannot|sorry|unable to|against policy|not able to)\b")
FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
IMAGE_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif",
}

SYSTEM_PROMPT_V2 = """You are labeling a private anime image dataset for a text-to-image LoRA.
Describe only what is visible. Do not moralize. Do not identify or name the artist.
Do not output `artist:` or `character:` tags; those are added deterministically by the pipeline.
Do not describe watermarks in captions; record them only in the watermark field.
Visible text must be copied exactly; never translate, romanize, or correct it.
Return one JSON object only, no Markdown."""

_FALLBACK_CAPTION_PROMPT = """You are an image recaptioner. Return exactly one JSON object with this schema:
{
  "caption": {
    "long": "a detailed image description, including all non-watermark visible text",
    "medium": "a one-paragraph image description",
    "short": "a one-sentence image description"
  },
  "watermark": [
    {"box_2d": [y_min, x_min, y_max, x_max], "label": "watermark text"}
  ],
  "borders": ["none"]
}

Rules:
- Describe only information visible in the image; preserve visible text exactly, excluding watermarks.
- Do not describe watermarks in any caption field; record them only in watermark.
- watermark.box_2d uses normalized 0-1000 [y_min, x_min, y_max, x_max] relative to the full image.
- borders uses left/right/up/down, or ["none"]; never combine "none" with a direction.
- All three caption fields must be non-empty strings.
- Return only the JSON object, without Markdown or explanations.

Here is the original caption reference:
{{reference}}"""

LAYOUT_CONTRACT = """\
Return the same caption/watermark/borders object and add a `layout` object:
{
  "layout": {
    "summary": {
      "meta": ["anime"],
      "canvas": "single",
      "caption": "global scene/style description"
    },
    "items": [
      {"label": "character", "caption": "visual appearance/pose/position",
       "bbox": [y_min, x_min, y_max, x_max]},
      {"label": "object", "caption": "object appearance",
       "bbox": [y_min, x_min, y_max, x_max]},
      {"label": "text", "caption": "typography/surface description",
       "text": "EXACT VISIBLE TEXT", "bbox": [y_min, x_min, y_max, x_max]}
    ],
    "bbox_coord_space": "normalized_1000",
    "bbox_format": "yxyx"
  }
}

Layout rules:
- bbox is [y_min, x_min, y_max, x_max] with integers 0-1000 relative to the full original image.
- label is exactly one of character, object, text.
- text items require the exact visible text; copy it without translation or correction.
- Use 1-12 items total and at most 8 text items. Group OCR lines into blocks/bubbles; do not emit every word.
- summary.canvas is single, multi_panel, or collage. summary.meta has at most 3 tags.
- Do not name the artist or a character in item captions; use "the character" or visual descriptions.
- If unsure, omit an item: a wrong item is worse than no item.
- watermark.box_2d uses the same normalized 0-1000 yxyx convention; [] if none.
- Do not output `artist:` or `character:` tags anywhere.
"""

LAYOUT_ONLY_PROMPT = """You are labeling the spatial layout of a private anime image for a text-to-image LoRA.
Return exactly one JSON object with a single `layout` key. Do not write captions.
Layout rules:
- bbox is [y_min, x_min, y_max, x_max] with integers 0-1000 relative to the full original image.
- label is exactly one of character, object, text.
- text items require the exact visible text; copy it without translation or correction.
- Use 1-12 items total and at most 8 text items. Group OCR lines into blocks/bubbles.
- summary.canvas is single, multi_panel, or collage. summary.meta has at most 3 tags.
- Do not name the artist or a character in item captions; use "the character" or visual descriptions.
- Use `bbox_coord_space`: "normalized_1000" and `bbox_format`: "yxyx".
- Do not output `artist:` or `character:` tags anywhere.
- Return only the JSON object, without Markdown or explanations.

OCR text/boxes are fixed context; copy their text exactly and do not invent text outside them:
{{ocr}}
"""


class VLMError(RuntimeError):
    """HTTP transport, response-shape or configuration failure."""


@dataclass
class VLMConfig:
    endpoint: str
    model: str
    api_key: str | None = None
    model_family: str = "auto"  # auto | gemma | qwen
    layout_mode: str = "combined"  # combined | caption | split
    timeout: float = 300.0
    max_attempts: int = MAX_ATTEMPTS
    max_http_retries: int = 1
    temperature: float = 1.0
    retry_temperature: float = 0.2
    top_p: float = 0.95
    retry_top_p: float = 0.9
    max_tokens: int = 16384
    layout_max_tokens: int = 4096
    thinking: bool | None = True
    image_first: bool | None = None
    response_format: bool = True
    use_system_prompt: bool = True
    prompt_path: str | None = None
    system_prompt_path: str | None = None
    system_prompt: str | None = None
    prompt_sha256: str = ""
    system_prompt_sha256: str = ""
    backoff_base: float = 2.0
    max_backoff: float = 30.0
    max_image_side: int | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise VLMError(f"endpoint must be an http(s) URL, got {self.endpoint!r}")
        if not self.model.strip():
            raise VLMError("model must be non-empty")
        if self.model_family not in ("auto", "gemma", "qwen"):
            raise VLMError(f"model_family must be auto/gemma/qwen, got {self.model_family!r}")
        if self.layout_mode not in ("combined", "caption", "split"):
            raise VLMError(
                f"layout_mode must be combined/caption/split, got {self.layout_mode!r}")
        if not 1 <= self.max_attempts <= MAX_ATTEMPTS:
            raise VLMError(f"max_attempts must be 1..{MAX_ATTEMPTS}")
        if self.timeout <= 0:
            raise VLMError("timeout must be positive")
        if self.max_http_retries < 0:
            raise VLMError("max_http_retries must be >= 0")
        if self.max_image_side is not None and self.max_image_side < 64:
            raise VLMError("max_image_side must be >= 64 or None")
        prompt_text = load_caption_prompt(self.prompt_path)
        system_text = resolve_system_prompt(self.system_prompt_path, self.system_prompt)
        self.prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        self.system_prompt_sha256 = hashlib.sha256(system_text.encode("utf-8")).hexdigest()

    @property
    def use_image_first(self) -> bool:
        if self.image_first is not None:
            return self.image_first
        name = f"{self.model_family} {self.model}".lower()
        if "gemma" in name:
            return True
        if "qwen" in name:
            return False
        return True

    def signature(self) -> str:
        """Stable hash of every request-shaping parameter (resume identity)."""
        payload = {
            "model": self.model,
            "model_family": self.model_family,
            "layout_mode": self.layout_mode,
            "temperature": self.temperature,
            "retry_temperature": self.retry_temperature,
            "top_p": self.top_p,
            "retry_top_p": self.retry_top_p,
            "max_tokens": self.max_tokens,
            "layout_max_tokens": self.layout_max_tokens,
            "thinking": self.thinking,
            "image_first": self.image_first,
            "response_format": self.response_format,
            "use_system_prompt": self.use_system_prompt,
            "max_image_side": self.max_image_side,
            "prompt_path": self.prompt_path,
            "prompt_sha256": self.prompt_sha256,
            "system_prompt_path": self.system_prompt_path,
            "system_prompt_sha256": self.system_prompt_sha256,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class VLMResult:
    status: str  # ok | repaired | refused | invalid
    model: str
    endpoint: str
    prompt_version: str
    prompt_sha256: str
    attempts: int
    http_retries: int
    latency_seconds: float
    route: str
    parsed: dict[str, Any] | None = None
    annotation: VLMAnnotation | None = None
    raw_content: str = ""
    refusal: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    fallback: bool = False

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "repaired") and self.annotation is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model,
            "endpoint": self.endpoint,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "attempts": self.attempts,
            "http_retries": self.http_retries,
            "latency_seconds": round(self.latency_seconds, 4),
            "route": self.route,
            "fallback": self.fallback,
            "refusal": self.refusal,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "usage": dict(self.usage),
            "annotation": self.parsed,
            "annotation_validated": self.annotation.to_dict() if self.annotation else None,
        }


def load_caption_prompt(path: str | Path | None = None) -> str:
    """Official ``docs/caption_prompt.txt`` with a packaged fallback."""
    candidate = Path(path) if path is not None else PROMPT_PATH
    try:
        text = candidate.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_CAPTION_PROMPT
    return text if "{{reference}}" in text else text + "\n\n{{reference}}"


def resolve_system_prompt(path: str | Path | None = None,
                          override: str | None = None) -> str:
    """Return the effective system prompt, preferring an explicit override."""
    if override is not None and override.strip():
        return override
    if path:
        try:
            text = Path(path).read_text(encoding="utf-8")
            if text.strip():
                return text
        except OSError:
            pass
    return SYSTEM_PROMPT_V2


def build_caption_prompt(reference: str, path: str | Path | None = None) -> str:
    return load_caption_prompt(path).replace("{{reference}}", reference.strip() or "(none)")


def build_caption_only_prompt(reference: str,
                              ocr_context: Sequence[Mapping[str, Any]] = (),
                              path: str | Path | None = None) -> str:
    """Official caption prompt, optionally with OCR text (no boxes) as context."""
    prompt = build_caption_prompt(reference, path).rstrip()
    context = format_ocr_text(ocr_context)
    if context:
        prompt += ("\n\nOCR text detected in the image (fixed context; copy it exactly, "
                   "do not invent text outside it):\n" + context)
    return prompt


def format_ocr_text(items: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for item in items:
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        lines.append(f"- {json.dumps(text, ensure_ascii=False)}")
        if len(lines) >= 20:
            break
    return "\n".join(lines)


def build_combined_prompt(reference: str, ocr_context: Sequence[Mapping[str, Any]] = (),
                          path: str | Path | None = None) -> str:
    parts = [build_caption_prompt(reference, path).rstrip(), LAYOUT_CONTRACT]
    if ocr_context:
        parts.append("OCR text/boxes are fixed context; copy their text exactly:\n"
                     + format_ocr_context(ocr_context))
    return "\n\n".join(parts)


def build_layout_prompt(ocr_context: Sequence[Mapping[str, Any]] = ()) -> str:
    return LAYOUT_ONLY_PROMPT.replace("{{ocr}}", format_ocr_context(ocr_context) or "(none)")


def format_ocr_context(items: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for item in items:
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        bbox = item.get("bbox")
        lines.append(f"- text={json.dumps(text, ensure_ascii=False)} bbox={json.dumps(bbox)}")
        if len(lines) >= 20:
            break
    return "\n".join(lines)


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def detect_refusal(text: str) -> str | None:
    match = REFUSAL_RE.search(text or "")
    return match.group(0) if match else None


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Extract the first parseable JSON object from prose/fenced/partial output."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty response"
    candidates: list[str] = [text.strip()]
    candidates.extend(match.group(1).strip() for match in FENCE_RE.finditer(text))
    candidates.extend(_balanced_candidates(text))
    errors: list[str] = []
    for candidate in candidates:
        for repaired in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError as error:
                errors.append(str(error))
                continue
            if isinstance(parsed, dict):
                return parsed, None
            errors.append("JSON value is not an object")
    return None, errors[-1] if errors else "no JSON object found"


def _balanced_candidates(text: str) -> list[str]:
    """Balanced-brace scans so leading/trailing prose does not hide the object."""
    spans: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    spans.append(text[start:index + 1])
                    break
    return spans


# --------------------------------------------------------------------------- HTTP


@dataclass
class HTTPResult:
    status: int
    body: str
    latency: float
    error: str | None = None


class VLMClient:
    """Synchronous OpenAI-compatible VLM client with retry/split/fallback."""

    def __init__(self, config: VLMConfig, fallback: VLMConfig | None = None):
        self.config = config
        self.fallback = fallback
        self.last_prompt = ""

    # -- images ------------------------------------------------------------

    def image_data_uri(self, image: str | Path | bytes | Image.Image,
                       mime: str | None = None) -> str:
        if isinstance(image, Image.Image):
            pil = ImageOps.exif_transpose(image).convert("RGB")
            raw, suffix = _encode_pil(pil, self.config.max_image_side)
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            payload = raw
        elif isinstance(image, (str, Path)):
            path = Path(image)
            payload = path.read_bytes()
            mime = mime or IMAGE_MIME.get(path.suffix.lower(), "image/png")
            if self.config.max_image_side:
                with Image.open(io.BytesIO(payload)) as opened:
                    pil = ImageOps.exif_transpose(opened).convert("RGB")
                    payload, suffix = _encode_pil(pil, self.config.max_image_side)
                    mime = "image/png" if suffix == ".png" else "image/jpeg"
        elif isinstance(image, (bytes, bytearray)):
            payload = bytes(image)
            mime = mime or "image/png"
        else:
            raise VLMError(f"unsupported image type: {type(image).__name__}")
        return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"

    # -- request building --------------------------------------------------

    def _messages(self, config: VLMConfig, prompt: str,
                  data_uri: str) -> list[dict[str, Any]]:
        image_part = {"type": "image_url", "image_url": {"url": data_uri}}
        text_part = {"type": "text", "text": prompt}
        content = ([image_part, text_part] if config.use_image_first
                   else [text_part, image_part])
        messages: list[dict[str, Any]] = []
        if config.use_system_prompt:
            messages.append({
                "role": "system",
                "content": resolve_system_prompt(
                    config.system_prompt_path, config.system_prompt),
            })
        messages.append({"role": "user", "content": content})
        return messages

    def _payload(self, config: VLMConfig, prompt: str, data_uri: str, *,
                 temperature: float, top_p: float, max_tokens: int,
                 thinking: bool | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": self._messages(config, prompt, data_uri),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if config.response_format:
            payload["response_format"] = {"type": "json_object"}
        if thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        payload.update(config.extra_payload)
        return payload

    def _headers(self, config: VLMConfig) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **config.extra_headers}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        return headers

    def _post(self, config: VLMConfig, payload: dict[str, Any]) -> HTTPResult:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(config.endpoint, data=data,
                                         headers=self._headers(config), method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                return HTTPResult(status=int(response.status), body=body,
                                  latency=time.perf_counter() - started)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            return HTTPResult(status=int(error.code), body=body,
                              latency=time.perf_counter() - started, error=str(error))
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as error:
            return HTTPResult(status=0, body="", latency=time.perf_counter() - started,
                              error=f"{type(error).__name__}: {error}")

    def _request(self, config: VLMConfig,
                 payload: dict[str, Any]) -> tuple[HTTPResult, int]:
        retries = 0
        result = HTTPResult(status=0, body="", latency=0.0, error="request not attempted")
        for attempt in range(config.max_http_retries + 1):
            result = self._post(config, payload)
            retryable = result.status == 0 or result.status == 429 or result.status >= 500
            if not retryable or attempt >= config.max_http_retries:
                return result, retries
            retries += 1
            self._sleep(config, attempt)
        return result, retries

    def _sleep(self, config: VLMConfig, attempt: int) -> None:
        delay = min(config.max_backoff, config.backoff_base * (2 ** attempt))
        if delay > 0:
            time.sleep(random.uniform(0, delay))

    # -- response parsing --------------------------------------------------

    def _call(self, config: VLMConfig, prompt: str, data_uri: str, *,
              temperature: float, top_p: float, max_tokens: int,
              thinking: bool | None
              ) -> tuple[dict[str, Any] | None, str, dict[str, int], int, float, list[str]]:
        self.last_prompt = prompt
        payload = self._payload(config, prompt, data_uri, temperature=temperature,
                                top_p=top_p, max_tokens=max_tokens, thinking=thinking)
        response, http_retries = self._request(config, payload)
        errors: list[str] = []
        usage: dict[str, int] = {}
        # Some OpenAI-compatible servers reject response_format rather than
        # ignoring it. Retry once without it and rely on JSON extraction.
        if response.status in (400, 422) and "response_format" in payload:
            payload.pop("response_format", None)
            retry, extra_retries = self._request(config, payload)
            http_retries += extra_retries
            if retry.status != response.status or retry.body != response.body:
                response = retry
                errors.append("server rejected response_format; retried without it")
        if response.error:
            errors.append(response.error)
        if not 200 <= response.status < 300:
            errors.append(f"HTTP {response.status}: {response.body[:500]}")
            return None, response.body, usage, http_retries, response.latency, errors
        try:
            body = json.loads(response.body)
        except json.JSONDecodeError as error:
            return None, response.body, usage, http_retries, response.latency, [
                f"invalid response JSON: {error}"]
        usage = _usage_from_body(body)
        try:
            content = _content_from_body(body)
        except VLMError as error:
            return None, response.body, usage, http_retries, response.latency, [str(error)]
        parsed, error = extract_json_object(content)
        if parsed is None:
            errors.append(error or "no JSON object")
        return parsed, content, usage, http_retries, response.latency, errors

    # -- validation --------------------------------------------------------

    def _validate_combined(self, parsed: Mapping[str, Any]) -> VLMAnnotation:
        return parse_vlm_annotation(parsed)

    def _validate_caption(self, parsed: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(parsed, Mapping):
            raise AnnotationError("caption response must be an object")
        if "caption" not in parsed and any(key in parsed for key in ("long", "medium", "short")):
            parsed = {"caption": parsed}
        caption = parse_raw_caption(parsed.get("caption"))
        watermark = parsed.get("watermark", [])
        borders = parsed.get("borders", ["none"])
        # Re-use combined validation for watermark/border/rating schema, without layout.
        scratch = {
            "caption": {"long": caption.long, "medium": caption.medium, "short": caption.short},
            "watermark": watermark,
            "borders": borders,
            "rating": parsed.get("rating"),
            "nsfw": parsed.get("nsfw"),
            "layout": {"summary": {"meta": ["anime"], "canvas": "single", "caption": "x"},
                       "items": [{"label": "object", "caption": "x", "bbox": [0, 0, 100, 100]}],
                       "bbox_coord_space": "normalized_1000", "bbox_format": "yxyx"},
        }
        validated = parse_vlm_annotation(scratch)
        result = {"caption": scratch["caption"], "watermark": watermark, "borders": borders}
        if validated.rating is not None:
            result["rating"] = validated.rating
        if validated.nsfw is not None:
            result["nsfw"] = validated.nsfw
        return result

    def _validate_layout(self, parsed: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(parsed, Mapping):
            raise AnnotationError("layout response must be an object")
        layout_raw = parsed.get("layout", parsed)
        return parse_layout(layout_raw).to_dict()

    def _validate_split(self, caption_raw: Mapping[str, Any],
                        layout_raw: Mapping[str, Any]) -> VLMAnnotation:
        merged = {**dict(caption_raw), "layout": layout_raw}
        return parse_vlm_annotation(merged)

    # -- public ------------------------------------------------------------

    def annotate_image(
        self,
        image: str | Path | bytes | Image.Image,
        reference: str,
        *,
        ocr_context: Sequence[Mapping[str, Any]] = (),
        source_size: tuple[int, int] | None = None,
        mime: str | None = None,
    ) -> VLMResult:
        del source_size  # normalized-1000 boxes do not need canvas dimensions
        data_uri = self.image_data_uri(image, mime=mime)
        configs = [(self.config, False)]
        if self.fallback is not None:
            configs.append((self.fallback, True))
        total_attempts = 0
        total_http_retries = 0
        total_latency = 0.0
        total_usage: dict[str, int] = {}
        all_errors: list[str] = []
        last_content = ""
        last_prompt_hash = ""
        last_model = self.config.model
        last_endpoint = self.config.endpoint
        saw_refusal: str | None = None

        for config, is_fallback in configs:
            for attempt in range(1, min(config.max_attempts, MAX_ATTEMPTS) + 1):
                total_attempts += 1
                caption_only = config.layout_mode == "caption"
                split = config.layout_mode == "split" or (
                    config.layout_mode == "combined" and attempt >= 3)
                thinking = config.thinking if attempt == 1 else False
                temperature = config.temperature if attempt == 1 else config.retry_temperature
                top_p = config.top_p if attempt == 1 else config.retry_top_p
                max_tokens = config.max_tokens
                if caption_only:
                    prompt = build_caption_only_prompt(
                        reference, ocr_context, path=config.prompt_path)
                else:
                    prompt = build_combined_prompt(
                        reference, ocr_context, path=config.prompt_path)
                if all_errors and attempt > 1:
                    prompt += ("\n\nPrevious response was invalid:\n- "
                               + "\n- ".join(all_errors[-6:])
                               + "\nReturn one corrected JSON object.")
                last_prompt_hash = prompt_sha256(prompt) if not split else last_prompt_hash
                last_model = config.model
                last_endpoint = config.endpoint
                parsed, content, usage, http_retries, latency, errors = self._call(
                    config, prompt, data_uri, temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens, thinking=thinking)
                total_http_retries += http_retries
                total_latency += latency
                for key, value in usage.items():
                    total_usage[key] = total_usage.get(key, 0) + value
                last_content = content
                if parsed is None:
                    refusal = detect_refusal(content)
                    if refusal:
                        saw_refusal = refusal
                        all_errors.append(f"refusal: {refusal}")
                    all_errors.extend(error for error in errors if error not in all_errors)
                    continue

                try:
                    if caption_only:
                        caption_raw = self._validate_caption(parsed)
                        layout = Layout(
                            summary_caption=caption_raw["caption"]["medium"],
                            items=(),
                            summary_meta=("anime",),
                            canvas="single",
                            bbox_coord_space=NORMALIZED_SPACE,
                            bbox_format=BOX_FORMAT,
                        )
                        parsed_combined = {
                            "caption": caption_raw["caption"],
                            "watermark": caption_raw["watermark"],
                            "borders": caption_raw["borders"],
                            "layout": layout.to_dict(),
                        }
                        for key in ("rating", "nsfw"):
                            if key in caption_raw:
                                parsed_combined[key] = caption_raw[key]
                        annotation = parse_vlm_annotation(
                            parsed_combined, require_layout=False)
                        route = "caption"
                    elif split:
                        caption_raw = self._validate_caption(parsed)
                        layout_prompt = build_layout_prompt(ocr_context)
                        last_prompt_hash = prompt_sha256(layout_prompt)
                        (layout_parsed, layout_content, layout_usage, layout_retries,
                         layout_latency, layout_errors) = self._call(
                            config, layout_prompt, data_uri, temperature=temperature,
                            top_p=top_p, max_tokens=config.layout_max_tokens, thinking=False)
                        total_http_retries += layout_retries
                        total_latency += layout_latency
                        for key, value in layout_usage.items():
                            total_usage[key] = total_usage.get(key, 0) + value
                        last_content = layout_content or last_content
                        if layout_parsed is None:
                            all_errors.extend(layout_errors or ["layout response has no JSON"])
                            continue
                        layout_raw = self._validate_layout(layout_parsed)
                        annotation = self._validate_split(caption_raw, layout_raw)
                        route = "split"
                        parsed_combined: dict[str, Any] = {
                            **caption_raw, "layout": layout_raw}
                    else:
                        annotation = self._validate_combined(parsed)
                        route = "combined"
                        parsed_combined = dict(parsed)
                except AnnotationError as error:
                    all_errors.append(str(error))
                    continue
                status = "ok" if (attempt == 1 and not is_fallback) else "repaired"
                return VLMResult(
                    status=status, model=config.model, endpoint=config.endpoint,
                    prompt_version=PROMPT_VERSION, prompt_sha256=last_prompt_hash,
                    attempts=total_attempts, http_retries=total_http_retries,
                    latency_seconds=total_latency, route=route,
                    parsed=parsed_combined, annotation=annotation,
                    raw_content=last_content, refusal=None, errors=all_errors,
                    warnings=list(annotation.warnings), usage=total_usage,
                    fallback=is_fallback,
                )

        return VLMResult(
            status="refused" if saw_refusal else "invalid",
            model=last_model, endpoint=last_endpoint, prompt_version=PROMPT_VERSION,
            prompt_sha256=last_prompt_hash, attempts=total_attempts,
            http_retries=total_http_retries, latency_seconds=total_latency,
            route="fallback" if self.fallback is not None else "primary",
            parsed=None, annotation=None, raw_content=last_content,
            refusal=saw_refusal, errors=all_errors, usage=total_usage,
            fallback=bool(self.fallback),
        )


def _encode_pil(image: Image.Image, max_side: int | None) -> tuple[bytes, str]:
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, round(image.width * scale)),
                              max(1, round(image.height * scale))), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), ".png"


def _usage_from_body(body: Mapping[str, Any]) -> dict[str, int]:
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        return {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            result[key] = value
    return result


def _content_from_body(body: Mapping[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VLMError("response has no choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise VLMError("response choice is not an object")
    message = choice.get("message")
    content: Any = None
    if isinstance(message, Mapping):
        content = message.get("content")
    if content is None:
        content = choice.get("text")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        content = "".join(parts)
    if not isinstance(content, str):
        raise VLMError("response message has no text content")
    return content


def reference_tags(tags: Sequence[str]) -> str:
    """Reference prose for expansion rows with no source prose."""
    clean = [str(tag).strip() for tag in tags if str(tag).strip()]
    if not clean:
        return "reference tags; no original prose"
    return "reference tags; no original prose: " + ", ".join(clean[:40])


def sanitize_reference(text: str) -> str:
    """Strip labels/handles from a reference string before prompting."""
    cleaned = strip_metadata_labels(text).text
    cleaned = re.sub(r"(?<![\w@])@[A-Za-z0-9_]+", "", cleaned)
    return re.sub(r"\s*,\s*", ", ", cleaned).strip(" ,;\n\t")
