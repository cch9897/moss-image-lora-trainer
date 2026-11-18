"""Caption-only VLM route and resume-identity regressions."""

from __future__ import annotations

import json

import pytest

from moss_lora.annotations import AnnotationError, parse_vlm_annotation
from moss_lora.vlm import LAYOUT_CONTRACT, VLMClient, VLMConfig
from tests.test_vlm import config as vlm_config, mock_server, response_body


def test_caption_mode_sends_no_layout_contract_and_returns_no_items():
    reference = "a girl under a red umbrella"
    content = json.dumps({
        "caption": {"long": "Long prose.", "medium": "Medium prose.", "short": "Short prose."},
        "watermark": [],
        "borders": ["none"],
    })
    with mock_server([response_body(content)]) as (url, server):
        result = VLMClient(vlm_config(url, layout_mode="caption")).annotate_image(
            b"fake-image", reference)
        assert result.status == "ok" and result.route == "caption"
        prompt = server.requests[0]["messages"][1]["content"][1]["text"]
        assert reference in prompt
        assert LAYOUT_CONTRACT not in prompt
        assert '"layout"' not in prompt and "bbox" not in prompt
        assert len(server.requests) == 1
        assert result.annotation is not None
        assert result.annotation.layout.items == ()


def test_caption_mode_preserves_captions_watermark_and_borders():
    caption = {
        "long": "A long caption describing composition, lighting, and framing.",
        "medium": "A medium one-paragraph caption.",
        "short": "A short one-sentence caption.",
    }
    watermark = [{"box_2d": [10, 20, 300, 400], "label": "COPYRIGHT SAMPLE"}]
    content = json.dumps({"caption": caption, "watermark": watermark,
                          "borders": ["left", "up"]})
    with mock_server([response_body(content)]) as (url, _server):
        result = VLMClient(vlm_config(url, layout_mode="caption")).annotate_image(
            b"fake-image", "reference")
        assert result.status == "ok" and result.route == "caption"
        annotation = result.annotation
        assert annotation is not None
        assert annotation.caption.long == caption["long"]
        assert annotation.caption.medium == caption["medium"]
        assert annotation.caption.short == caption["short"]
        assert [mark.to_dict() for mark in annotation.watermark] == watermark
        assert annotation.borders.to_list() == ["left", "up"]


def test_config_signature_is_stable_and_tracks_request_shaping():
    values = dict(endpoint="http://127.0.0.1:9/v1/chat/completions",
                  model="gemma-4-31b-test", model_family="gemma")
    first = VLMConfig(**values, layout_mode="caption")
    identical = VLMConfig(**values, layout_mode="caption")
    assert first.signature() == identical.signature()
    assert VLMConfig(**values, layout_mode="caption",
                     temperature=0.5).signature() != first.signature()
    assert VLMConfig(**values, layout_mode="combined").signature() != first.signature()


def test_empty_layout_items_require_explicit_relaxation():
    layout = {
        "summary": {"meta": ["anime"], "canvas": "single", "caption": "caption-only frame"},
        "items": [],
        "bbox_coord_space": "normalized_1000",
        "bbox_format": "yxyx",
    }
    payload = {
        "caption": {"long": "Long.", "medium": "Medium.", "short": "Short."},
        "watermark": [],
        "borders": ["none"],
        "layout": layout,
    }
    annotation = parse_vlm_annotation(payload, require_layout=False)
    assert annotation.layout.items == ()
    with pytest.raises(AnnotationError):
        parse_vlm_annotation(payload)
    with pytest.raises(AnnotationError):
        parse_vlm_annotation({**payload, "layout": {**layout, "items": None}},
                             require_layout=False)
