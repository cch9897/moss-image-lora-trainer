#!/usr/bin/env python3
"""Build the cleanV2 caption/metadata manifest for MOSS-Image LoRA training.

cleanV2 makes three changes to a v1-clean30 / v4-finalize metadata.jsonl:

1. ``short`` becomes a true tag-only variant: trigger + artist/character prefix
   + up to 12 general tags, no prose.
2. Generic style prose is removed from every prose-bearing field with a
   conservative clause-level filter (leading media frame, pure-style sentences,
   style-only comma clauses, lighting/palette/technique clauses, boilerplate
   quality words).  Quoted text, NSFW terms, visible text and content nouns are
   protected.  ``caption_no_artist`` keeps its prose (it is never reduced to a
   tag-only caption).
3. Artists below the target (median by default) are upsampled to exactly the
   target rows per epoch.  Head artists keep all of their rows.  Unknown /
   no-artist rows stay at 1x.  Repeats are appended after every original row, so
   the original cache item order (and any ``sample_val_ids``) stays valid.

The script is standalone (stdlib only) and writes::

    <output>/metadata.jsonl
    <output>/report.json
    <output>/README.md

Usage::

    python scripts/make_clean_v2.py \
        --metadata /path/to/metadata.jsonl \
        --output /path/to/cleanV2 \
        [--images-root /path/to/dataset] [--limit N] [--target 74]

Absolute ``image`` paths are written for the cache JSONL, so the output JSONL
may live outside the image tree (e.g. ~/dataset/cleanV2) without
symlinks.  The cache builder resolves ``image`` relative to the JSONL parent,
so absolute paths are required.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DATASET_VERSION = "cleanV2"
DEFAULT_TRIGGER = "@CH3OCH3"
LONG_TAG_LIMIT = 40
MEDIUM_TAG_LIMIT = 25
SHORT_TAG_LIMIT = 12
UNKNOWN_ARTISTS = {"", "unknown", "none", "n/a", "null"}

# ---------------------------------------------------------------------------
# Style vocabulary
# ---------------------------------------------------------------------------

# Tokens allowed inside a removable clause.  Deliberately excludes content
# nouns, colours with a concrete value, NSFW terms, and ambiguous medium words
# such as monochrome / sketch / watercolor / manga / anime figure.
_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "by", "with", "and", "or",
    "featuring", "featured", "rendered", "drawn", "depicted", "painted",
    "illustrated", "shown", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "from", "for", "to",
    "into", "over", "under", "against", "high", "highly", "very", "extremely",
    "blend", "mixed", "slight", "slightly", "subtle", "subtly", "strong",
    "strongly", "delicate",
}
_STYLE = {
    # lighting / palette
    "soft", "warm", "cool", "dramatic", "cinematic", "volumetric", "rim",
    "studio", "ambient", "vibrant", "muted", "pastel", "rich", "limited",
    "simple", "pale", "bold", "colorful", "lighting", "light", "lit",
    "bathed", "glow", "glowing", "dusk", "dawn", "daylight", "sunlight",
    "moonlight", "color", "colour", "colors", "colours", "palette", "palettes",
    "tones", "tone", "scheme", "saturated", "desaturated", "shading", "shaded",
    # quality / boilerplate
    "quality", "best", "masterpiece", "8k", "4k", "intricate", "ultra",
    "hyper", "detailed", "detail", "professional", "award", "winning",
    # media / technique
    "anime", "anime-style", "digital", "illustration", "artwork", "image",
    "drawing", "render", "rendering", "piece", "painting", "cel", "shade",
    "flat", "lineart", "line", "art", "lines", "linework", "ink", "outline",
    "painterly", "3d", "cgi", "realistic", "semi", "semirealistic", "clean",
    "crisp", "sharp", "smooth", "stylized", "style",
    # Adjectives and finish vocabulary seen in generic style boilerplate.
    # Clause removal still requires a strong style token and only stop/style
    # words, so content clauses (``bright blue eyes``) are never removed.
    "bright", "minimalist", "elegant", "brushwork", "brush", "stroke",
    "strokes", "highlight", "highlights", "contrast", "texture", "finish",
    "hues", "gradient", "gradients", "saturation", "cel-shaded",
    "cel-shading", "line-art", "flat-colored", "flat-coloured",
    "semi-realistic",
}
# A clause is only removable if it contains at least one of these words: this
# prevents deleting incidental stop-word-only fragments.
_STRONG_STYLE = {
    "lighting", "light", "lit", "glow", "glowing", "shading", "shaded",
    "palette", "tones", "colors", "color", "colour", "colours", "illustration",
    "artwork", "image", "render", "rendering", "drawing", "painting", "style",
    "painterly", "lineart", "linework", "cel", "digital", "anime", "manga",
    "vibrant", "muted", "pastel", "soft", "warm", "cool", "clean", "flat",
}
# Additional tokens permitted only when the *whole sentence* is style/mood
# boilerplate (e.g. "Soft lighting and a pale color palette create a clean,
# intimate atmosphere.").  Mood/composition are kept in normal clauses.
_PURE_EXTRA = {
    "scene", "picture", "composition", "characterized", "marked", "defined",
    "create", "creates", "creating", "make", "makes", "evoke", "evokes",
    "evoking", "atmosphere", "mood", "intimate", "serene", "cozy", "beauty",
    "beautiful", "aesthetically", "aesthetic", "show", "shows", "showing",
}
_COLOR_WORDS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "grey", "gray", "gold", "golden", "silver",
}
_NSFW_TERMS = {
    "nipples", "nipple", "areola", "areolae", "penis", "vagina", "vulva",
    "anus", "anal", "sex", "cum", "semen", "ejaculation", "masturbation",
    "bondage", "bdsm", "nude", "naked", "nudity", "topless", "bottomless",
    "pubic", "genitals", "genital", "crotch", "breasts", "breast", "cleavage",
    "panties", "underwear", "panty", "pussy", "cock", "dick", "fellatio",
    "oral", "vaginal", "penetration", "erect", "erection", "sperm", "orgasm",
    "clitoris", "labia", "scrotum", "testicles", "buttocks", "asshole", "butt",
    "ass",
}
# Finite verbs make a comma clause a complete statement; removing it would
# leave a fragment (e.g. "The lighting is soft and warm, reflecting ...").
_FINITE_VERBS = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "creates", "create", "makes", "make", "shows", "show", "features",
    "feature", "stands", "stand", "sits", "sit", "lies", "lie", "hangs",
    "hang", "covers", "cover", "fills", "fill", "reflects", "reflect",
    "illuminates", "illuminate", "frames", "frame", "gives", "give",
}
_QUOTE_CHARS = "\"“”"

# ---------------------------------------------------------------------------
# Conservative strip regexes (from docs/caption_deconfounding_plan.md §2.4)
# ---------------------------------------------------------------------------

_LEADING_MEDIUM = re.compile(
    r"(?i)^\s*(?:an?|the)\s+"
    r"(?:(?:highly|very|extremely)\s+)?"
    r"(?:(?:detailed|digital|anime[- ]style|anime|manga|cel[- ]shaded|"
    r"painterly|watercolou?r|3d|cgi|realistic|flat[- ]colou?red|"
    r"vibrant|bright|colorful|colourful|saturated|minimalist|soft|warm|cool|"
    r"elegant|stylized|stylised|intricate|highly detailed|professional|"
    r"semi[- ]realistic)\s+){1,6}"
    r"(?:illustration|artwork|image|drawing|render|piece)\s+of\s+"
)
_STYLE_CLAUSE = re.compile(
    r"(?i),\s*(?:rendered|drawn|depicted|painted|illustrated|shown)\s+in\s+"
    r"[^.;]{0,90}?\b(?:style|manner|technique|medium|rendering)\b"
)
_LIGHT_PALETTE_CLAUSE = re.compile(
    r"(?i),\s*(?:with|featuring)\s+(?:a\s+)?"
    r"(?:soft|warm|cool|dramatic|cinematic|vibrant|muted|pastel|rich|"
    r"limited|simple|pale|bold|colorful)\s+(?:[a-z-]+\s+){0,3}?"
    r"(?:lighting|colou?r palette|palette|colou?rs?|tones?)\b(?!\s+of\b)[^.;]*"
)
_TECHNIQUE_CLAUSE = re.compile(
    r"(?i),\s*(?:featuring|with)\s+(?:a\s+)?(?:blend of\s+)?"
    r"(?:clean lines?(?:\s+and\s+soft shading)?|soft shading|cel shading|"
    r"line ?art|linework|flat colou?rs?)\b[^.;]*"
)
_QUALITY_WORDS = re.compile(
    r"(?i)\b(?:masterpiece|best quality|high[- ]quality|ultra[- ]detailed|"
    r"hyper[- ]?detailed|highly detailed|intricate details?|8k|4k)\b,?\s*"
)
_PURE_STYLE_SENTENCE = re.compile(
    r"(?i)^(?:the\s+)?(?:scene|image|picture|artwork|illustration|composition)\s+"
    r"(?:is\s+)?(?:characterized|marked|defined)\s+by\s+(?:a\s+)?"
    r"(?:soft|warm|cool|dramatic|cinematic|vibrant|muted|pastel|rich|limited|"
    r"simple|pale|bold|colorful|clean)\s+(?:[a-z-]+\s+){0,3}?"
    r"(?:lighting|colou?r palette|palette|colou?rs?|tones?)"
    r"(?:\s+and\s+(?:a\s+)?(?:soft|warm|cool|vibrant|muted|pastel|rich|limited|"
    r"simple|pale)\s+(?:[a-z-]+\s+){0,3}?(?:lighting|palette|colou?rs?|tones?))?"
    r"\s*[.!?]?$"
)
# Whole-sentence style subject statements: ``The art style features clean line
# art, soft cel shading, and bright saturated colors.``  The predicate must
# contain only style/stop vocabulary; a content noun or finite content verb
# keeps the sentence (e.g. ``Soft warm dusk lighting illuminates the dark
# wooden shoji doors.``).
_STYLE_SUBJECT_SENTENCE = re.compile(
    r"(?i)^\s*(?:the\s+)?(?:art\s+style|artstyle|overall\s+style|visual\s+style|"
    r"style|rendering|lighting|palette|colou?r\s+palette|colou?r\s+scheme|"
    r"colou?rs?|shading|line\s?art|linework|technique)\s+"
    r"(?:is|are|was|were|features|feature|uses|use|showcases|showcase|shows|"
    r"show|employs|employ)\s+"
)


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return [w.casefold() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)]


def _has_nsfw(text: str) -> bool:
    low = text.casefold()
    return any(
        re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", low)
        for term in _NSFW_TERMS
    )


def _has_protected_text(text: str, visible: Sequence[str]) -> bool:
    return any(value and len(value) >= 2 and value in text for value in visible)


def _split_outside_quotes(text: str) -> list[tuple[str, str]]:
    """Split into (unquoted|quoted, segment) runs; quote spans are preserved.

    Double quotes, curly quotes and single-quoted phrases (opened only at a
    word boundary, so apostrophes such as ``girl's`` are not treated as quotes)
    are protected.
    """
    parts: list[tuple[str, str]] = []
    buf: list[str] = []
    quote_char: str | None = None
    index = 0
    while index < len(text):
        ch = text[index]
        if quote_char is not None:
            buf.append(ch)
            if ch == quote_char:
                parts.append(("q", "".join(buf)))
                buf = []
                quote_char = None
        elif ch == '"':
            if buf:
                parts.append(("u", "".join(buf)))
                buf = []
            quote_char = '"'
            buf.append(ch)
        elif ch == "“":
            if buf:
                parts.append(("u", "".join(buf)))
                buf = []
            quote_char = "”"
            buf.append(ch)
        elif ch == "'":
            previous = text[index - 1] if index else " "
            closing = text.find("'", index + 1)
            boundary = previous.isspace() or previous in "([{—–-.;:!?\"“”," 
            if boundary and closing > index + 1:
                if buf:
                    parts.append(("u", "".join(buf)))
                    buf = []
                quote_char = "'"
            buf.append(ch)
        else:
            buf.append(ch)
        index += 1
    if buf:
        parts.append(("q" if quote_char is not None else "u", "".join(buf)))
    return parts


def _apply_outside_quotes(text: str, regex: re.Pattern[str],
                          visible: Sequence[str]) -> tuple[str, int]:
    removed = 0
    pieces: list[str] = []
    for kind, segment in _split_outside_quotes(text):
        if kind == "u":
            def replacement(match: re.Match[str]) -> str:
                nonlocal removed
                if _deletable_span(match.group(0), visible):
                    removed += 1
                    return ""
                return match.group(0)

            segment = regex.sub(replacement, segment)
        pieces.append(segment)
    return "".join(pieces), removed


def _has_quoted_span(text: str) -> bool:
    return any(kind == "q" for kind, _ in _split_outside_quotes(text))


def _deletable_span(span: str, visible: Sequence[str]) -> bool:
    if _has_quoted_span(span):
        return False
    if _has_nsfw(span) or _has_protected_text(span, visible):
        return False
    words = _tokenize(span)
    if not words:
        return False
    return all(word in _STYLE or word in _STOP for word in words)


def _is_style_clause(clause: str, visible: Sequence[str]) -> bool:
    if _has_quoted_span(clause):
        return False
    if _has_nsfw(clause) or _has_protected_text(clause, visible):
        return False
    words = _tokenize(clause)
    if not words or not (set(words) & _STRONG_STYLE):
        return False
    if set(words) & _FINITE_VERBS:
        return False
    return all(word in _STYLE or word in _STOP for word in words)


def _is_pure_style_sentence(sentence: str, visible: Sequence[str]) -> bool:
    if _has_quoted_span(sentence):
        return False
    if _has_nsfw(sentence) or _has_protected_text(sentence, visible):
        return False
    stripped = sentence.strip()
    if _PURE_STYLE_SENTENCE.match(stripped):
        return True
    match = _STYLE_SUBJECT_SENTENCE.match(stripped)
    if match:
        predicate = stripped[match.end():]
        predicate_words = _tokenize(predicate)
        if (predicate_words
                and not (set(predicate_words) & _FINITE_VERBS)
                and all(word in _STYLE or word in _STOP or word in _PURE_EXTRA
                        or word in _COLOR_WORDS for word in predicate_words)):
            return True
    words = _tokenize(stripped)
    if not words or not (set(words) & _STRONG_STYLE):
        return False
    return all(
        word in _STYLE or word in _STOP or word in _PURE_EXTRA or word in _COLOR_WORDS
        for word in words
    )


def _last_word(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)
    return words[-1].casefold() if words else ""


def _clause_continues(previous_clause: str) -> bool:
    """True if the previous clause ends in a word that expects a continuation."""
    last = _last_word(previous_clause)
    return bool(last) and (last in _STYLE or last in _STOP)


def _split_clauses(text: str) -> list[str]:
    clauses: list[str] = []
    buf: list[str] = []
    for kind, segment in _split_outside_quotes(text):
        if kind == "q":
            buf.append(segment)
            continue
        pieces = segment.split(",")
        for position, piece in enumerate(pieces):
            if position > 0:
                clauses.append("".join(buf))
                buf = []
            buf.append(piece)
    clauses.append("".join(buf))
    return clauses


def _split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation outside quotes; keep decimals/abbrevs."""
    sentences: list[str] = []
    start = 0
    in_quote = False
    for i, ch in enumerate(text):
        if ch == '"':
            in_quote = not in_quote
        if in_quote or ch not in ".!?":
            continue
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if nxt and not nxt.isspace():
            continue
        prev = text[:i].rstrip()
        if prev and (prev[-1].isdigit() or re.search(
                r"(?i)\b(?:mr|mrs|ms|dr|st|vs|etc|e\.g|i\.e)$", prev)):
            continue
        sentences.append(text[start:i + 1].strip())
        start = i + 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences or [text.strip()]


def _cleanup(text: str, final_punct: str = "") -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r",\s*([.!?])", r"\1", text)
    text = re.sub(r"\s+([.!?])", r"\1", text)
    text = re.sub(
        r"(?i)\b(and|with|by|of|in|featuring|including)\s*([.!?])", r"\2", text)
    text = re.sub(r"^[\s,;]+", "", text).strip()
    # Conservative article repair after removing a leading frame.
    text = re.sub(
        r"(?i)\bA\s+(anime|image|illustration|artwork|outline|ink|orange)\b",
        r"An \1", text)
    text = re.sub(
        r"(?i)\bAn\s+(girl|woman|boy|man|character|person|figure|scene|"
        r"background|dress|detailed|highly|digital|anime-style)\b",
        r"A \1", text)
    text = re.sub(
        r"(?i)\b(?:A|An)\s+one\s+(girl|woman|boy|man|character|person|figure)\b",
        r"A \1", text)
    text = re.sub(
        r"(?i)^one\s+(girl|woman|boy|man|character|person|figure)\b",
        r"a \1", text)
    if text:
        text = text[0].upper() + text[1:]
        if final_punct and not re.search(r"[.!?]+$", text):
            text += final_punct
    return text


# ---------------------------------------------------------------------------
# Prose strip
# ---------------------------------------------------------------------------


def _strip_sentence(sentence: str, visible: Sequence[str]) -> tuple[str, dict[str, int]]:
    original = sentence
    stats = {"sentences_removed": 0, "clauses_removed": 0}
    if not sentence.strip():
        return sentence, stats

    if _is_pure_style_sentence(sentence, visible):
        stats["sentences_removed"] = 1
        return "", stats

    match = _LEADING_MEDIUM.match(sentence)
    if match and _deletable_span(match.group(0), visible):
        sentence = sentence[match.end():]
        # The frame often says "of one girl ..."; once it is gone, use the
        # natural indefinite article.
        sentence = re.sub(
            r"(?i)^(?:one|1)\s+(girl|woman|boy|man|character|person|figure)\b",
            r"a \1", sentence)
        stats["clauses_removed"] += 1
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]

    clauses = _split_clauses(sentence)
    leading_removed = 0
    index = 0
    while index < len(clauses) - 1 and _is_style_clause(clauses[index], visible):
        leading_removed += 1
        index += 1
    rest = clauses[index:]
    kept: list[str] = []
    for position, clause in enumerate(rest):
        if (position > 0 and _is_style_clause(clause, visible)
                and not _clause_continues(kept[-1] if kept else "")):
            stats["clauses_removed"] += 1
            continue
        kept.append(clause)
    if leading_removed:
        first = kept[0].lstrip() if kept else ""
        prefix = "An image, " if first and first[0].islower() else ""
        sentence = prefix + ", ".join(part.strip() for part in kept)
        stats["clauses_removed"] += leading_removed
    else:
        sentence = ", ".join(part.strip() for part in kept)

    for regex in (_STYLE_CLAUSE, _LIGHT_PALETTE_CLAUSE, _TECHNIQUE_CLAUSE,
                  _QUALITY_WORDS):
        sentence, removed = _apply_outside_quotes(sentence, regex, visible)
        stats["clauses_removed"] += removed

    if not stats["clauses_removed"] and not stats["sentences_removed"]:
        return original, stats

    final_punct = "".join(re.findall(r"[.!?]+$", original))
    sentence = _cleanup(sentence, final_punct)
    return sentence, stats


def strip_prose(prose: str, visible: Sequence[str] = (),
                *, allow_empty: bool = True) -> tuple[str, dict[str, Any]]:
    """Apply the conservative style strip to one prose field.

    Returns ``(new_prose, stats)``.  Quoted spans are never modified and must
    survive verbatim; if that invariant would be violated the original prose is
    returned.  When ``allow_empty`` is false (caption_no_artist) an empty result
    falls back to the original prose so the variant never becomes tag-only.
    """
    stats: dict[str, Any] = {
        "changed": False,
        "chars_before": len(prose),
        "chars_after": len(prose),
        "sentences_removed": 0,
        "clauses_removed": 0,
        "emptied": 0,
        "fallback": 0,
        "quote_guard": 0,
    }
    if not prose.strip():
        return prose, stats

    original_quotes = []
    for kind, segment in _split_outside_quotes(prose):
        if kind == "q" and len(segment) > 2:
            original_quotes.append(segment)

    output_sentences: list[str] = []
    for sentence in _split_sentences(prose):
        stripped, sentence_stats = _strip_sentence(sentence, visible)
        if stripped:
            output_sentences.append(stripped)
        stats["sentences_removed"] += sentence_stats["sentences_removed"]
        stats["clauses_removed"] += sentence_stats["clauses_removed"]

    result = re.sub(r"\s+", " ", " ".join(output_sentences)).strip()

    # Quoted text must survive verbatim.
    if any(quote not in result for quote in original_quotes):
        stats["fallback"] = 1
        stats["quote_guard"] = 1
        return prose, stats

    if not re.search(r"[A-Za-z]{2}", result):
        if allow_empty:
            stats["emptied"] = 1
            stats["changed"] = bool(prose.strip())
            stats["chars_after"] = 0
            return "", stats
        stats["fallback"] = 1
        return prose, stats

    stats["changed"] = result != prose
    stats["chars_after"] = len(result)
    return result, stats


# ---------------------------------------------------------------------------
# Regression cases (exact misses found by the independent audit)
# ---------------------------------------------------------------------------

REGRESSION_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "A vibrant anime-style digital illustration of one girl standing under "
        "cherry blossoms.",
        "A girl standing under cherry blossoms.",
        "leading medium frame with 'vibrant' and 'one girl' normalization",
    ),
    (
        "The art style features clean line art, soft cel shading, and bright "
        "saturated colors.",
        "",
        "pure-style subject sentence (line art / cel shading / saturated colors)",
    ),
    (
        "The art style features soft painterly brushwork and a muted warm "
        "color palette.",
        "",
        "pure-style subject sentence (painterly brushwork / warm palette)",
    ),
    (
        "Soft warm dusk lighting illuminates the dark wooden shoji doors.",
        "Soft warm dusk lighting illuminates the dark wooden shoji doors.",
        "content sentence kept (lighting illuminates wooden doors)",
    ),
    (
        "The art style features a glowing sword.",
        "The art style features a glowing sword.",
        "content noun in the predicate keeps the sentence",
    ),
    (
        "The art style features a nude figure.",
        "The art style features a nude figure.",
        "NSFW term keeps the sentence",
    ),
    (
        "The rendering is soft and warm, illuminating the 'OPEN' sign.",
        "The rendering is soft and warm, illuminating the 'OPEN' sign.",
        "quoted visible text keeps the sentence",
    ),
)


def run_regression_cases() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    failed = 0
    for source, expected, note in REGRESSION_CASES:
        actual, _ = strip_prose(source)
        passed = actual == expected
        failed += 0 if passed else 1
        cases.append({
            "input": source,
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "note": note,
        })
    return {"passed": failed == 0, "failed": failed, "cases": cases}


# ---------------------------------------------------------------------------
# Row-level caption reconstruction
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def row_artists(row: Mapping[str, Any], *, include_unknown: bool = False) -> list[str]:
    artists = _dedupe(_as_list(row.get("artists")))
    if not artists:
        artist = row.get("artist")
        if isinstance(artist, str) and artist.strip():
            artists = [artist.strip()]
    if include_unknown:
        return artists
    return [value for value in artists if value.casefold() not in UNKNOWN_ARTISTS]


def row_characters(row: Mapping[str, Any]) -> list[str]:
    return _dedupe(_as_list(row.get("character")))


def row_tags(row: Mapping[str, Any]) -> list[str]:
    return _dedupe(_as_list(row.get("tags")))


def artist_key(row: Mapping[str, Any]) -> str | None:
    artists = row_artists(row)
    if artists:
        return artists[0]
    raw = row.get("artist")
    if isinstance(raw, str) and raw.strip().casefold() not in UNKNOWN_ARTISTS:
        return raw.strip()
    return None


def build_prefix(trigger: str, artists: Sequence[str], characters: Sequence[str],
                 *, include_artist: bool = True) -> list[str]:
    parts = [trigger]
    if include_artist:
        parts.extend(f"artist:{artist}" for artist in artists)
    parts.extend(f"character:{character}" for character in characters)
    return _dedupe(parts)


def compose_caption(prefix: Sequence[str], tags: Sequence[str], prose: str = "",
                    *, max_tags: int) -> str:
    parts = list(prefix) + list(tags[:max_tags])
    text = ", ".join(part for part in parts if part)
    prose = prose.strip()
    if prose:
        text = f"{text}, {prose}" if text else prose
    return text


def split_tagged_caption(
    text: str,
    *,
    trigger: str,
    artists: Sequence[str],
    characters: Sequence[str],
    tags: Sequence[str],
    include_artist: bool = True,
) -> tuple[int, str]:
    """Return ``(tag_count, prose)`` for one prefix+tags+prose string.

    The tag section is consumed from the start: first the canonical prefix
    tokens, then consecutive segments that match ``tags``.  Prose is everything
    after that boundary and is returned byte-identically (joined with ", ").
    """
    if not text or not text.strip():
        return 0, ""
    prefix_keys = {part.casefold() for part in build_prefix(
        trigger, artists, characters, include_artist=include_artist)}
    tag_keys = {tag.casefold() for tag in tags}
    segments = text.split(", ")
    index = 0
    while index < len(segments) and segments[index].strip().casefold() in prefix_keys:
        index += 1
    tag_count = 0
    while index < len(segments) and segments[index].strip().casefold() in tag_keys:
        index += 1
        tag_count += 1
    prose = ", ".join(segments[index:]).strip()
    return tag_count, prose


def _looks_like_prose(text: str) -> bool:
    """Layout item ``caption`` is usually a short category label, not prose."""
    words = re.findall(r"[A-Za-z]{2,}", text)
    return len(words) >= 4 or bool(re.search(r"[.!?]", text))


def _visible_texts(layout: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(layout, Mapping):
        return []
    values: list[str] = []
    for item in layout.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        if isinstance(text, str) and len(text.strip()) >= 2:
            values.append(text.strip())
    return values


def _field_stats_merge(target: dict[str, Any], stats: Mapping[str, Any]) -> None:
    target["fields"] += 1
    if stats.get("changed"):
        target["changed"] += 1
    target["sentences_removed"] += int(stats.get("sentences_removed", 0))
    target["clauses_removed"] += int(stats.get("clauses_removed", 0))
    target["emptied"] += int(stats.get("emptied", 0))
    target["fallback"] += int(stats.get("fallback", 0))
    target["quote_guard"] += int(stats.get("quote_guard", 0))
    target["chars_before"] += int(stats.get("chars_before", 0))
    target["chars_after"] += int(stats.get("chars_after", 0))


def _new_field_stats() -> dict[str, Any]:
    return {
        "fields": 0, "changed": 0, "sentences_removed": 0,
        "clauses_removed": 0, "emptied": 0, "fallback": 0,
        "quote_guard": 0, "chars_before": 0, "chars_after": 0,
    }


def _caption_sources(row: Mapping[str, Any]) -> dict[str, str | None]:
    caption = row.get("caption")
    sources: dict[str, str | None] = {
        "long": None, "medium": None, "short": None,
    }
    if isinstance(caption, Mapping):
        for key in sources:
            value = caption.get(key)
            if isinstance(value, str) and value.strip():
                sources[key] = value
    elif isinstance(caption, str) and caption.strip():
        for key in sources:
            sources[key] = caption
    original = row.get("caption_original")
    sources["original"] = original if isinstance(original, str) and original.strip() else None
    no_artist = row.get("caption_no_artist")
    sources["no_artist"] = no_artist if isinstance(no_artist, str) and no_artist.strip() else None
    layout = row.get("layout")
    if isinstance(layout, Mapping):
        summary = layout.get("summary")
        if isinstance(summary, Mapping):
            value = summary.get("caption")
            if isinstance(value, str) and value.strip():
                sources["layout"] = value
    return sources


def _images_root_default(metadata_path: Path) -> Path:
    return metadata_path.parent


def resolve_image(row: Mapping[str, Any], images_root: Path) -> tuple[str, bool]:
    """Absolute image path; prefers an existing source_path, else file_name."""
    source_path = row.get("source_path")
    if isinstance(source_path, str) and source_path.strip():
        candidate = Path(source_path).expanduser()
        if not candidate.is_absolute():
            candidate = images_root / candidate
        candidate = candidate.resolve()
        if candidate.is_file():
            return str(candidate), True
    file_name = row.get("file_name")
    if isinstance(file_name, str) and file_name.strip():
        candidate = Path(file_name).expanduser()
        if not candidate.is_absolute():
            candidate = images_root / candidate
        candidate = candidate.resolve()
        return str(candidate), candidate.is_file()
    return "", False


def transform_row(
    row: Mapping[str, Any],
    *,
    trigger: str,
    images_root: Path,
    stats: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Transform one row; returns (new_row, image_exists)."""
    new_row = copy.deepcopy(dict(row))
    artists = row_artists(row)
    characters = row_characters(row)
    tags = row_tags(row)
    visible = _visible_texts(row.get("layout"))

    prefix = build_prefix(trigger, artists, characters, include_artist=True)
    no_artist_prefix = build_prefix(trigger, artists, characters, include_artist=False)
    sources = _caption_sources(row)
    # Missing structured fields fall back to the surviving caption text rather
    # than silently dropping prose.
    if not sources.get("long"):
        sources["long"] = (sources.get("original") or sources.get("layout")
                           or sources.get("medium"))
    if not sources.get("medium"):
        sources["medium"] = sources.get("long")
    if not sources.get("original"):
        sources["original"] = sources.get("long") or sources.get("layout")

    def parse(field: str, *, include_artist: bool = True) -> tuple[int, str]:
        text = sources.get(field)
        if not text:
            return 0, ""
        return split_tagged_caption(
            text, trigger=trigger, artists=artists, characters=characters,
            tags=tags, include_artist=include_artist)

    long_tags, long_prose = parse("long")
    medium_tags, medium_prose = parse("medium")
    short_tags, short_prose = parse("short")
    original_tags, original_prose = parse("original")
    no_artist_tags, no_artist_prose = parse("no_artist", include_artist=False)
    layout_tags, layout_prose = parse("layout")

    # Tag counts follow the source variant where it had a tag section.  If a
    # legacy row has no tag section, use the historical cap.
    long_tags = min(long_tags or LONG_TAG_LIMIT, len(tags))
    medium_tags = min(medium_tags or MEDIUM_TAG_LIMIT, len(tags))
    original_tags = min(original_tags or LONG_TAG_LIMIT, len(tags))
    no_artist_tags = min(no_artist_tags or LONG_TAG_LIMIT, len(tags))
    layout_tags = min(layout_tags or LONG_TAG_LIMIT, len(tags))

    def stripped(field: str, prose: str, *, allow_empty: bool) -> str:
        result, field_stats = strip_prose(prose, visible, allow_empty=allow_empty)
        _field_stats_merge(stats[field], field_stats)
        return result

    long_text = compose_caption(
        prefix, tags, stripped("long", long_prose, allow_empty=True),
        max_tags=long_tags)
    medium_text = compose_caption(
        prefix, tags, stripped("medium", medium_prose, allow_empty=True),
        max_tags=medium_tags)
    short_text = compose_caption(prefix, tags, "", max_tags=SHORT_TAG_LIMIT)
    original_text = compose_caption(
        prefix, tags, stripped("original", original_prose, allow_empty=True),
        max_tags=original_tags)
    no_artist_text: str | None = None
    if sources.get("no_artist"):
        no_artist_text = compose_caption(
            no_artist_prefix, tags,
            stripped("no_artist", no_artist_prose, allow_empty=False),
            max_tags=no_artist_tags)

    if isinstance(new_row.get("caption"), Mapping):
        new_row["caption"]["long"] = long_text
        new_row["caption"]["medium"] = medium_text
        new_row["caption"]["short"] = short_text
    else:
        new_row["caption"] = {
            "long": long_text, "medium": medium_text, "short": short_text,
        }
    new_row["caption_original"] = original_text
    if no_artist_text is not None:
        new_row["caption_no_artist"] = no_artist_text

    # Layout summary + item captions.  `text` fields are never touched.
    layout = new_row.get("layout")
    if isinstance(layout, Mapping):
        summary = layout.get("summary")
        if isinstance(summary, Mapping) and isinstance(summary.get("caption"), str):
            summary["caption"] = compose_caption(
                prefix, tags,
                stripped("layout", layout_prose, allow_empty=True),
                max_tags=layout_tags)
        for item in layout.get("items") or []:
            if not isinstance(item, Mapping):
                continue
            caption_value = item.get("caption")
            if (not isinstance(caption_value, str)
                    or not caption_value.strip()
                    or not _looks_like_prose(caption_value)):
                continue
            result, item_stats = strip_prose(caption_value, visible, allow_empty=False)
            _field_stats_merge(stats["layout_item"], item_stats)
            if result:
                item["caption"] = result

    new_row["dataset_version"] = DATASET_VERSION
    image, exists = resolve_image(row, images_root)
    if image:
        new_row["image"] = image
    return new_row, exists


# ---------------------------------------------------------------------------
# Balancing / report / validation
# ---------------------------------------------------------------------------


def compute_balance(
    rows: Sequence[Mapping[str, Any]], target: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Append upsample-only duplicates after all base rows."""
    base = [copy.deepcopy(dict(row)) for row in rows]
    counts = Counter(key for row in rows if (key := artist_key(row)))
    artist_positions: dict[str, int] = defaultdict(int)
    repeats: list[dict[str, Any]] = []
    copies_histogram: Counter[int] = Counter()
    repeat_index_histogram: Counter[int] = Counter()

    for row in base:
        key = artist_key(row)
        count = counts.get(key or "", 0)
        if not key or not count or count >= target:
            continue
        position = artist_positions[key]
        artist_positions[key] += 1
        quotient, remainder = divmod(target, count)
        copies = quotient + (1 if position < remainder else 0)
        copies_histogram[copies] += 1
        for repeat_index in range(1, copies):
            duplicate = copy.deepcopy(row)
            duplicate["repeat_of"] = row.get("file_name")
            duplicate["repeat_index"] = repeat_index
            repeat_index_histogram[repeat_index] += 1
            repeats.append(duplicate)

    output = base + repeats
    balance = {
        "base_rows": len(base),
        "balanced_rows": len(output),
        "extra_rows": len(repeats),
        "target": target,
        "artist_counts_base": dict(sorted(counts.items())),
        "artists_total": len(counts),
        "artists_boosted": sum(1 for count in counts.values() if count < target),
        "artist_counts_balanced": dict(sorted(Counter(
            key for row in output if (key := artist_key(row))).items())),
        "copies_histogram": {str(k): v for k, v in sorted(copies_histogram.items())},
        "repeat_index_histogram": {
            str(k): v for k, v in sorted(repeat_index_histogram.items())},
        "repeats": repeats,
    }
    return output, balance


def _all_caption_texts(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    caption = row.get("caption")
    if isinstance(caption, Mapping):
        values.extend(str(caption.get(key) or "") for key in ("long", "medium", "short"))
    elif isinstance(caption, str):
        values.append(caption)
    for key in ("caption_original", "caption_no_artist"):
        value = row.get(key)
        if isinstance(value, str):
            values.append(value)
    layout = row.get("layout")
    if isinstance(layout, Mapping):
        summary = layout.get("summary")
        if isinstance(summary, Mapping) and isinstance(summary.get("caption"), str):
            values.append(summary["caption"])
    return values


def _caption_variant_text(row: Mapping[str, Any], field: str) -> str:
    caption = row.get("caption")
    if field in ("long", "medium", "short"):
        if isinstance(caption, Mapping):
            return str(caption.get(field) or "")
        if isinstance(caption, str):
            return caption
    if field == "original":
        return str(row.get("caption_original") or "")
    if field == "no_artist":
        return str(row.get("caption_no_artist") or "")
    if field == "layout":
        layout = row.get("layout")
        if isinstance(layout, Mapping):
            summary = layout.get("summary")
            if isinstance(summary, Mapping):
                return str(summary.get("caption") or "")
    return ""


def validate(
    base_rows: Sequence[Mapping[str, Any]],
    output_rows: Sequence[Mapping[str, Any]],
    *,
    trigger: str,
    target: int,
    balance: Mapping[str, Any],
    expected_balanced: int,
) -> tuple[list[str], int]:
    errors: list[str] = []
    base_count = len(base_rows)

    if len(output_rows) != expected_balanced:
        errors.append(
            f"balanced row count {len(output_rows)} != expected {expected_balanced}")
    if balance.get("balanced_rows") != expected_balanced:
        errors.append("balance report row count disagrees with formula")

    # Original rows first, in original order, no repeat markers.
    for index in range(base_count):
        if index >= len(output_rows):
            errors.append("output truncated before all original rows")
            break
        expected_name = base_rows[index].get("file_name")
        actual_name = output_rows[index].get("file_name")
        if actual_name != expected_name:
            errors.append(
                f"base row {index} reordered: {actual_name!r} != {expected_name!r}")
            break
        if "repeat_of" in output_rows[index]:
            errors.append(f"base row {index} carries repeat_of")
            break

    boosted = {
        artist for artist, count in balance["artist_counts_base"].items()
        if count < target
    }
    balanced_counts = Counter(
        key for row in output_rows if (key := artist_key(row)))
    for artist in boosted:
        if balanced_counts.get(artist, 0) < target:
            errors.append(
                f"artist {artist!r} has {balanced_counts.get(artist, 0)} < {target} rows")

    empty_caption_count = 0
    missing_artist_tag_count = 0
    tag_only_short_count = 0
    non_absolute_images = 0
    quote_loss_rows = 0

    # Compare quoted spans in the original base rows with the emitted base rows.
    for source, emitted in zip(base_rows, output_rows[:base_count]):
        for field in ("long", "medium", "original", "no_artist", "layout"):
            before = _caption_sources(source).get(field)
            if not before:
                continue
            after = _caption_variant_text(emitted, field)
            for quote in (segment for kind, segment in _split_outside_quotes(before)
                          if kind == "q" and len(segment) > 2):
                if quote not in after:
                    quote_loss_rows += 1
                    if quote_loss_rows <= 5:
                        errors.append(
                            f"{source.get('file_name')}: {field} lost quoted text "
                            f"{quote!r}")

    for row in output_rows:
        texts = _all_caption_texts(row)
        if not any(text.strip() for text in texts):
            empty_caption_count += 1

        image = row.get("image")
        if not isinstance(image, str) or not Path(image).is_absolute():
            non_absolute_images += 1

        artists = row_artists(row)
        tags = row_tags(row)
        prefix = build_prefix(
            trigger, artists, row_characters(row), include_artist=True)
        expected_short = compose_caption(
            prefix, tags, "", max_tags=SHORT_TAG_LIMIT)
        actual_short = _caption_variant_text(row, "short")
        if actual_short != expected_short:
            # Count only once, not per repeated duplicate.
            if row.get("repeat_of") is None:
                errors.append(
                    f"{row.get('file_name')}: short is not tag-only/expected")
        else:
            tag_only_short_count += 1

        if artists:
            for artist in artists:
                token = f"artist:{artist}"
                for field in ("long", "medium", "original", "layout"):
                    text = _caption_variant_text(row, field)
                    if not text.strip():
                        continue
                    # Tag-only variants are exempt (no prose after the tag section).
                    _, prose = split_tagged_caption(
                        text, trigger=trigger, artists=artists,
                        characters=row_characters(row), tags=tags,
                        include_artist=True)
                    if prose and token not in text:
                        missing_artist_tag_count += 1
                        if row.get("repeat_of") is None:
                            errors.append(
                                f"{row.get('file_name')}: {field} missing {token}")

    if empty_caption_count:
        errors.append(f"{empty_caption_count} rows have no non-empty caption")
    if missing_artist_tag_count:
        errors.append(
            f"{missing_artist_tag_count} prose variants missing their artist tag")
    if tag_only_short_count != len(output_rows):
        errors.append(
            f"tag-only short count {tag_only_short_count} != {len(output_rows)}")
    if non_absolute_images:
        errors.append(f"{non_absolute_images} rows have a non-absolute image path")
    if quote_loss_rows:
        errors.append(f"{quote_loss_rows} quoted spans were lost during stripping")
    return errors, quote_loss_rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    tmp.replace(path)


def write_readme(path: Path, report: Mapping[str, Any], *, metadata: Path,
                 output: Path, images_root: Path | None) -> None:
    counts = report["counts"]
    artists = report["artists"]
    strip = report["strip"]
    lines = [
        "# cleanV2 dataset manifest",
        "",
        f"Generated by `scripts/make_clean_v2.py` from `{metadata}`.",
        f"Dataset version: `{DATASET_VERSION}`.",
        "",
        "## Counts",
        "",
        f"- Base rows: **{counts['base_rows']:,}**",
        f"- Balanced rows: **{counts['balanced_rows']:,}**",
        f"- Appended repeats: **{counts['extra_rows']:,}**",
        f"- Artist target (median): **{artists['target']}**",
        f"- Known artists: **{artists['known']:,}**; unknown/no-artist rows: "
        f"**{artists['unknown_rows']:,}**",
        f"- Artists boosted: **{artists['boosted']:,}**",
        f"- True tag-only `short` rows (base): **{counts['tag_only_short_base']:,}** "
        f"(balanced: **{counts['tag_only_short_balanced']:,}**)",
        f"- Empty captions: **{counts['empty_captions']}**",
        "",
        "## Variants",
        "",
        "- `caption.short`: trigger + artist/character prefix + up to 12 tags, no prose.",
        "- `caption.long` / `caption.medium`: stripped prose, same tag caps as the "
        "source (40 / 25 by default).",
        "- `caption_original`: tag-only in v4 (preserved); v1 prose is stripped.",
        "- `caption_no_artist`: artist tag removed, prose retained and stripped; "
        "never reduced to tag-only.",
        "- `layout.summary.caption`: stripped prose; layout item `caption` fields are "
        "stripped and `text` fields are never modified.",
        "",
        "## Balancing",
        "",
        "Upsample-only. For each known artist with `c < target`:",
        "",
        "```text",
        "q, r = divmod(target, c)",
        "copies(image j) = q + (1 if j < r else 0)",
        "```",
        "",
        "Each boosted artist contributes exactly `target` rows per epoch. Head "
        "artists keep every row; unknown/no-artist rows stay at 1x. Repeats are "
        "appended after all base rows, so original cache item IDs remain valid.",
        "",
        "## Paths",
        "",
        f"Output JSONL: `{output / 'metadata.jsonl'}`",
        f"Images root: `{images_root}`",
        "Every row carries an absolute `image` path (cache JSONL requirement).",
        "",
        "## Style strip",
        "",
        "| field | fields | changed | rate | emptied | fallback | chars removed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for field, values in strip.items():
        fields = values["fields"]
        rate = (values["changed"] / fields * 100) if fields else 0.0
        removed = values["chars_before"] - values["chars_after"]
        lines.append(
            f"| `{field}` | {fields:,} | {values['changed']:,} | {rate:.1f}% | "
            f"{values['emptied']:,} | {values['fallback']:,} | {removed:,} |")
    lines.extend([
        "",
        "The filter is clause-level only. It never globally deletes `anime`, "
        "`detailed`, `monochrome`, `sketch`, `watercolor`, `pastel`, or "
        "`anime figure`; quoted text, NSFW terms, visible text, colours with a "
        "concrete value, and content nouns block a clause from being removed.",
        "",
        "## Validation",
        "",
        f"- Status: **{'PASS' if report['validation']['passed'] else 'FAIL'}**",
        f"- Errors: {len(report['validation']['errors'])}",
        "",
    ])
    if report["validation"]["errors"]:
        lines.extend(f"- {error}" for error in report["validation"]["errors"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({error})") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def detect_trigger(rows: Sequence[Mapping[str, Any]], override: str | None) -> str:
    if override:
        return override if override.startswith("@") else f"@{override}"
    for row in rows:
        for field in ("caption", "caption_original", "caption_no_artist"):
            value = row.get(field)
            if isinstance(value, Mapping):
                value = value.get("long") or value.get("medium") or value.get("short")
            if isinstance(value, str) and value.strip():
                token = value.split(",", 1)[0].strip()
                if token.startswith("@"):
                    return token
    return DEFAULT_TRIGGER


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the cleanV2 balanced, style-stripped metadata JSONL.")
    parser.add_argument("--metadata", required=True, type=Path,
                        help="Input metadata.jsonl (v1 clean30 or v4 finalize schema)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for metadata.jsonl/report.json/README.md")
    parser.add_argument("--images-root", type=Path, default=None,
                        help="Root for relative file_name paths (default: metadata parent)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only read the first N base rows (dry-run aid)")
    parser.add_argument("--target", type=int, default=None,
                        help="Artist row target per epoch (default: median of known artists)")
    parser.add_argument("--trigger", default=None,
                        help=f"Trigger token (default: auto-detect, else {DEFAULT_TRIGGER})")
    parser.add_argument("--examples", type=int, default=0,
                        help="Capture N before/after strip examples in report.json")
    parser.add_argument("--no-strict", action="store_true",
                        help="Write outputs even if validation fails")
    args = parser.parse_args(argv)

    metadata = args.metadata.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not metadata.is_file():
        parser.error(f"metadata file not found: {metadata}")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    images_root = (args.images_root.expanduser().resolve()
                   if args.images_root else _images_root_default(metadata))
    if not images_root.is_dir():
        parser.error(f"images root is not a directory: {images_root}")

    rows = read_jsonl(metadata, args.limit)
    trigger = detect_trigger(rows, args.trigger)
    artist_counts = Counter(key for row in rows if (key := artist_key(row)))
    if args.target is not None:
        if args.target <= 0:
            parser.error("--target must be positive")
        target = args.target
        median_value = statistics.median(artist_counts.values()) if artist_counts else 0
    else:
        median_value = statistics.median(artist_counts.values()) if artist_counts else 0
        target = max(1, int(math.floor(median_value + 0.5))) if artist_counts else 0

    strip_stats = {
        name: _new_field_stats()
        for name in ("long", "medium", "original", "no_artist", "layout", "layout_item")
    }
    examples: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    missing_images = 0

    for row in rows:
        new_row, exists = transform_row(
            row, trigger=trigger, images_root=images_root, stats=strip_stats)
        if not exists:
            missing_images += 1
        base_rows.append(new_row)

        if len(examples) < args.examples:
            sources = _caption_sources(row)
            artists = row_artists(row)
            characters = row_characters(row)
            tags = row_tags(row)
            for field in ("long", "medium", "original", "no_artist", "layout"):
                before = sources.get(field)
                if not before:
                    continue
                after = _caption_variant_text(new_row, field)
                if after and after != before:
                    _, before_prose = split_tagged_caption(
                        before, trigger=trigger, artists=artists,
                        characters=characters, tags=tags,
                        include_artist=field != "no_artist")
                    _, after_prose = split_tagged_caption(
                        after, trigger=trigger, artists=artists,
                        characters=characters, tags=tags,
                        include_artist=field != "no_artist")
                    if not after_prose.strip():
                        continue
                    examples.append({
                        "file_name": row.get("file_name"),
                        "artist": row.get("artist"),
                        "field": field,
                        "prose_before": before_prose,
                        "prose_after": after_prose,
                        "prose_chars_removed": len(before_prose) - len(after_prose),
                    })
                    if len(examples) >= args.examples:
                        break

    balanced_rows, balance = compute_balance(base_rows, target)
    expected_balanced = len(base_rows) + sum(
        max(0, target - count) for count in artist_counts.values())
    errors, quoted_loss_rows = validate(
        rows, balanced_rows, trigger=trigger, target=target, balance=balance,
        expected_balanced=expected_balanced)
    regressions = run_regression_cases()
    if not regressions["passed"]:
        errors.append(
            f"{regressions['failed']} style-strip regression case(s) failed")

    tag_only_short = 0
    tag_only_short_base = 0
    empty_captions = 0
    for row in balanced_rows:
        artists = row_artists(row)
        expected_short = compose_caption(
            build_prefix(trigger, artists, row_characters(row)), row_tags(row), "",
            max_tags=SHORT_TAG_LIMIT)
        if _caption_variant_text(row, "short") == expected_short:
            tag_only_short += 1
        if not any(text.strip() for text in _all_caption_texts(row)):
            empty_captions += 1
    for row in base_rows:
        artists = row_artists(row)
        expected_short = compose_caption(
            build_prefix(trigger, artists, row_characters(row)), row_tags(row), "",
            max_tags=SHORT_TAG_LIMIT)
        if _caption_variant_text(row, "short") == expected_short:
            tag_only_short_base += 1

    known_artists = artist_counts
    unknown_rows = len(rows) - sum(known_artists.values())
    strip_report: dict[str, Any] = {}
    for field, values in strip_stats.items():
        report_values = dict(values)
        report_values["changed_rate"] = (
            values["changed"] / values["fields"] if values["fields"] else 0.0)
        report_values["chars_removed"] = values["chars_before"] - values["chars_after"]
        strip_report[field] = report_values

    short_prose_dropped = 0
    for row in rows:
        _, prose = split_tagged_caption(
            _caption_sources(row).get("short") or "", trigger=trigger,
            artists=row_artists(row), characters=row_characters(row),
            tags=row_tags(row))
        short_prose_dropped += len(prose)

    report = {
        "dataset_version": DATASET_VERSION,
        "input": str(metadata),
        "output": str(output),
        "images_root": str(images_root),
        "trigger": trigger,
        "limit": args.limit,
        "counts": {
            "base_rows": len(base_rows),
            "balanced_rows": len(balanced_rows),
            "extra_rows": balance["extra_rows"],
            "expected_balanced_rows": expected_balanced,
            "tag_only_short_base": tag_only_short_base,
            "tag_only_short_balanced": tag_only_short,
            "empty_captions": empty_captions,
            "missing_images": missing_images,
            "short_prose_chars_dropped": short_prose_dropped,
        },
        "artists": {
            "known": len(known_artists),
            "unknown_rows": unknown_rows,
            "median": median_value,
            "target": target,
            "boosted": balance["artists_boosted"],
            "counts_base": balance["artist_counts_base"],
            "counts_balanced": balance["artist_counts_balanced"],
            "copies_histogram": balance["copies_histogram"],
            "repeat_index_histogram": balance["repeat_index_histogram"],
        },
        "strip": strip_report,
        "regressions": regressions,
        "examples": examples,
        "validation": {
            "passed": not errors,
            "errors": errors,
            "quoted_loss_rows": quoted_loss_rows,
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "metadata.jsonl", balanced_rows)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output / "README.md", report, metadata=metadata,
                 output=output, images_root=images_root)

    print(f"base rows       : {len(base_rows):,}")
    print(f"balanced rows   : {len(balanced_rows):,} (expected {expected_balanced:,})")
    print(f"extra rows      : {balance['extra_rows']:,}")
    print(f"artist target   : {target} (median {median_value})")
    print(f"artists boosted : {balance['artists_boosted']:,}")
    print(f"tag-only short  : base {tag_only_short_base:,}, balanced {tag_only_short:,}")
    print(f"empty captions  : {empty_captions}")
    print(f"missing images  : {missing_images}")
    for field, values in strip_report.items():
        print(f"strip {field:11s}: {values['changed']:,}/{values['fields']:,} "
              f"({values['changed_rate'] * 100:.1f}%), removed "
              f"{values['chars_removed']:,} chars")
    print(f"validation      : {'PASS' if not errors else 'FAIL'} "
          f"({len(errors)} error(s))")
    print(f"regressions     : {len(REGRESSION_CASES) - regressions['failed']}/"
          f"{len(REGRESSION_CASES)} passed")
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    if errors and not args.no_strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
