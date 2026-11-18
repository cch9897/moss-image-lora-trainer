"""VLM client tests against a mock OpenAI-compatible HTTP server (no model)."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from moss_lora.vlm import (
    PROMPT_VERSION,
    VLMClient,
    VLMConfig,
    VLMError,
    build_combined_prompt,
    detect_refusal,
    extract_json_object,
)


def combined_content() -> str:
    payload = {
        "caption": {
            "long": "A girl stands in a city. artist:alice should be stripped.",
            "medium": "A girl in a city.",
            "short": "A girl outdoors.",
        },
        "watermark": [],
        "borders": ["none"],
        "layout": {
            "summary": {"meta": ["anime"], "canvas": "single", "caption": "city scene"},
            "items": [
                {"label": "character", "caption": "the character", "bbox": [100, 100, 800, 800]},
                {"label": "text", "caption": "sign", "text": "EXIT", "bbox": [500, 500, 600, 650]},
            ],
            "bbox_coord_space": "normalized_1000",
            "bbox_format": "yxyx",
        },
    }
    return json.dumps(payload)


def caption_content() -> str:
    return json.dumps({
        "caption": {"long": "Long prose.", "medium": "Medium prose.", "short": "Short prose."},
        "watermark": [], "borders": ["none"],
    })


def layout_content() -> str:
    return json.dumps({
        "layout": {
            "summary": {"meta": ["anime"], "canvas": "single", "caption": "city"},
            "items": [{"label": "object", "caption": "a sign", "bbox": [100, 100, 300, 300]}],
            "bbox_coord_space": "normalized_1000",
            "bbox_format": "yxyx",
        }
    })


def response_body(content: str, *, status: int = 200) -> tuple[int, str]:
    return status, json.dumps({
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    })


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(payload)  # type: ignore[attr-defined]
        responses = self.server.responses  # type: ignore[attr-defined]
        if not responses:
            status, body = 500, '{"error":"no response queued"}'
        else:
            status, body = responses.pop(0)
        if status == 0:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error":"forced"}')
            return
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args):  # noqa: D401, ARG002
        return


@contextmanager
def mock_server(responses):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests = []  # type: ignore[attr-defined]
    server.responses = list(responses)  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}/v1/chat/completions", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def config(url: str, **overrides) -> VLMConfig:
    values = dict(endpoint=url, model="gemma-4-31b-test", model_family="gemma",
                  timeout=2.0, max_attempts=3, max_http_retries=1,
                  backoff_base=0.0, max_backoff=0.0, thinking=True)
    values.update(overrides)
    return VLMConfig(**values)


def test_message_order_image_first_for_gemma_and_text_first_for_qwen():
    with mock_server([response_body(combined_content())]) as (url, server):
        result = VLMClient(config(url)).annotate_image(b"fake-image", "reference")
        assert result.status == "ok" and result.annotation is not None
        content = server.requests[0]["messages"][1]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["type"] == "text"
        assert server.requests[0]["chat_template_kwargs"] == {"enable_thinking": True}

    with mock_server([response_body(combined_content())]) as (url, server):
        result = VLMClient(config(url, model="qwen3.8-27b-test", model_family="qwen")
                          ).annotate_image(b"fake-image", "reference")
        assert result.status == "ok"
        content = server.requests[0]["messages"][1]["content"]
        assert content[0]["type"] == "text" and content[1]["type"] == "image_url"


def test_fenced_and_prose_wrapped_json_is_extracted():
    wrapped = "Here you go:\n```json\n" + combined_content() + "\n```\nDone."
    with mock_server([response_body(wrapped)]) as (url, server):
        result = VLMClient(config(url)).annotate_image(b"x", "ref")
        assert result.status == "ok"
        assert server.requests[0]["response_format"] == {"type": "json_object"}


def test_extract_json_object_repairs_trailing_commas_and_prose():
    parsed, error = extract_json_object('prefix {"a": 1,} suffix')
    assert parsed == {"a": 1} and error is None
    parsed, error = extract_json_object('{"a": {"b": 2}} trailing')
    assert parsed == {"a": {"b": 2}} and error is None


def test_refusal_retries_then_succeeds():
    with mock_server([
        response_body("I'm sorry, I can't help with that."),
        response_body(combined_content()),
    ]) as (url, server):
        result = VLMClient(config(url)).annotate_image(b"x", "ref")
        assert result.status == "repaired"
        assert result.attempts == 2
        assert result.refusal is None
        assert "Previous response was invalid" in server.requests[1]["messages"][1]["content"][1]["text"]
        assert server.requests[1]["temperature"] == 0.2
        assert result.annotation.caption.long == "A girl stands in a city."


def test_all_refusals_report_refused():
    with mock_server([response_body("I cannot comply.")] * 3) as (url, _server):
        result = VLMClient(config(url)).annotate_image(b"x", "ref")
        assert result.status == "refused"
        assert result.refusal and "cannot" in result.refusal.lower()
        assert result.annotation is None


def test_split_caption_layout_retry():
    with mock_server([
        response_body('{"caption": {}}'),
        response_body('{"caption": {}}'),
        response_body(caption_content()),
        response_body(layout_content()),
    ]) as (url, server):
        result = VLMClient(config(url)).annotate_image(b"x", "ref")
        assert result.status == "repaired" and result.route == "split"
        assert result.attempts == 3
        assert result.parsed["layout"]["items"][0]["label"] == "object"
        assert "Return exactly one JSON object with a single `layout` key" in (
            server.requests[3]["messages"][1]["content"][1]["text"])


def test_http_500_is_retried_once():
    with mock_server([
        (500, '{"error":"boom"}'),
        response_body(combined_content()),
    ]) as (url, _server):
        result = VLMClient(config(url)).annotate_image(b"x", "ref")
        assert result.status == "ok"
        assert result.http_retries == 1
        assert result.usage["total_tokens"] == 30


def test_fallback_model_is_used_after_primary_fails():
    with mock_server([
        response_body('{"caption": {}}'),
        response_body('{"caption": {}}'),
        response_body('{"caption": {}}'),
        response_body(combined_content()),
    ]) as (url, server):
        primary = config(url, model="gemma-test", model_family="gemma", max_attempts=3)
        fallback = config(url, model="qwen-test", model_family="qwen", max_attempts=3)
        result = VLMClient(primary, fallback).annotate_image(b"x", "ref")
        assert result.status == "repaired" and result.fallback is True
        assert result.model == "qwen-test"
        assert server.requests[-1]["messages"][1]["content"][0]["type"] == "text"


def test_annotation_validation_strips_artist_leak_and_watermark_text():
    payload = json.loads(combined_content())
    payload["watermark"] = [{"box_2d": [0, 0, 100, 1000], "label": "@artist"}]
    payload["caption"]["long"] = "A girl. artist:alice, character:hero."
    with mock_server([response_body(json.dumps(payload))]) as (url, _server):
        result = VLMClient(config(url)).annotate_image(b"x", "ref")
        assert result.status == "ok"
        assert "artist:" not in result.annotation.caption.long
        assert "character:" not in result.annotation.caption.long
        assert any("vlm_label_leak" in warning for warning in result.warnings)


def test_invalid_config_does_not_download():
    with pytest.raises(VLMError):
        VLMConfig(endpoint="ftp://x", model="m")
    with pytest.raises(VLMError):
        VLMConfig(endpoint="http://x", model="m", max_attempts=4)


def test_build_prompt_uses_official_docs_and_reference():
    prompt = build_combined_prompt("a reference", [{"text": "EXIT", "bbox": [1, 2, 3, 4]}])
    assert "image recaptioner" in prompt
    assert "a reference" in prompt and "layout" in prompt and "EXIT" in prompt
    assert detect_refusal("I'm sorry, I cannot") is not None
