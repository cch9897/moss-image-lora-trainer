"""Cache v2 variants: five kinds, layout separation, weights, legacy indexes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from moss_lora.cache import (
    CaptionVariant,
    build_cache,
    caption_variant_records,
    caption_variants,
    normalize_variant_weights,
    parse_variant_weights,
)
from moss_lora.dataset import CachedDataset
from moss_lora.utils import sha256_text
from tests.helpers import FakeTextEncoder, fake_latent_encoder_factory, gradient_image


def layout_dict(caption: str = "A poster.") -> dict:
    return {
        "summary": {"meta": ["anime"], "canvas": "single", "caption": caption},
        "items": [{"label": "text", "caption": "visible text", "text": "EXIT",
                   "bbox": [-100, -100, -50, 50]}],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


def final_row() -> dict:
    return {
        "caption_original": "@CH3OCH3, artist:alice, 1girl, source prose",
        "caption": {
            "long": "@CH3OCH3, artist:alice, 1girl, long prose",
            "medium": "@CH3OCH3, artist:alice, 1girl, medium prose",
            "short": "@CH3OCH3, artist:alice, 1girl, short prose",
        },
        "layout": layout_dict(),
    }


def test_caption_variant_records_five_kinds_layout_separate():
    records = caption_variant_records(final_row())
    assert [record.kind for record in records] == [
        "original", "long", "medium", "short", "layout"]
    assert [record.text for record in records[:4]] == [
        "@CH3OCH3, artist:alice, 1girl, source prose",
        "@CH3OCH3, artist:alice, 1girl, long prose",
        "@CH3OCH3, artist:alice, 1girl, medium prose",
        "@CH3OCH3, artist:alice, 1girl, short prose",
    ]
    layout = json.loads(records[-1].text)
    assert layout["summary"]["caption"] == "A poster."
    assert "long prose" not in records[-1].text  # layout never replaces caption.long


def test_caption_variant_dedupes_identical_texts_and_keeps_original():
    records = caption_variant_records({
        "caption_original": "same",
        "caption": {"long": "same", "medium": "same", "short": "same"},
    })
    assert records == [CaptionVariant("original", "same")]
    assert caption_variants({"caption": {"long": "a", "medium": "a"}}) == ["a"]


def test_layout_duplicate_key_order_is_deduped():
    layout = layout_dict()
    reordered = {"bbox_format": "yxyx", "items": layout["items"],
                 "bbox_coord_space": "abs_centralized",
                 "summary": layout["summary"]}
    records = caption_variant_records({
        "caption_original": "original",
        "caption": {"long": json.dumps(layout), "medium": "medium prose",
                    "short": "short prose"},
        "layout": reordered,
    })
    assert [record.kind for record in records] == ["original", "medium", "short", "layout"]
    # Reordered layout would serialize differently without canonical dedupe.
    assert json.loads(records[-1].text) == reordered


def test_variant_weight_normalization_and_override():
    kinds = ["original", "long", "medium", "short", "layout"]
    weights = normalize_variant_weights(kinds)
    assert weights == pytest.approx([0.16, 0.64 / 3, 0.64 / 3, 0.64 / 3, 0.20])
    assert sum(weights) == pytest.approx(1.0)
    assert normalize_variant_weights(["long", "medium", "short"]) == pytest.approx([1 / 3] * 3)
    assert normalize_variant_weights(["original"]) == [1.0]
    override = normalize_variant_weights(kinds, {"long": 2.0, "medium": 1.0})
    assert override == [0.0, 2 / 3, 1 / 3, 0.0, 0.0]
    assert normalize_variant_weights(["long"], {"medium": 1.0}) == [1.0]  # all-zero fallback
    with pytest.raises(ValueError):
        normalize_variant_weights(["long"], {"long": -1})
    assert parse_variant_weights("long=0.5, medium=0.5") == {"long": 0.5, "medium": 0.5}
    with pytest.raises(ValueError):
        parse_variant_weights("long")


def _build(tmp_path: Path, **kwargs) -> Path:
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    gradient_image(256, 256, 0).save(root / "img.png")
    manifest = root / "meta.jsonl"
    manifest.write_text(json.dumps({"image": "img.png", **final_row()}) + "\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    build_cache(
        manifest, cache_dir,
        text_encoder_factory=FakeTextEncoder,
        latent_encoder_factory=fake_latent_encoder_factory(8),
        target_area=256 * 256, progress=False, device="cpu",
        max_text_length=kwargs.pop("max_text_length", 1024), **kwargs)
    return cache_dir


def test_cache_build_five_variants_and_index(tmp_path):
    cache_dir = _build(tmp_path)
    index = json.loads((cache_dir / "index.json").read_text())
    item = index["items"][0]
    assert item["variant_kinds"] == ["original", "long", "medium", "short", "layout"]
    assert item["variant_weights"] == pytest.approx([0.16, 0.64 / 3, 0.64 / 3, 0.64 / 3, 0.20])
    assert len(item["text_lengths"]) == 5
    assert len(item["caption_sha256"]) == 5
    assert item["caption_sha256"] == [
        sha256_text(record.text) for record in caption_variant_records(final_row())]
    assert item["text_lengths"][0] >= 3
    assert index["max_text_length"] == 1024


def test_cache_variant_weight_override_and_max_text_length(tmp_path):
    cache_dir = _build(tmp_path, variant_weights={"long": 1.0}, max_text_length=4096)
    index = json.loads((cache_dir / "index.json").read_text())
    item = index["items"][0]
    assert index["max_text_length"] == 4096
    assert item["variant_weights"] == [0.0, 1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="max_text_length"):
        _build(tmp_path / "bad", max_text_length=4097)


def test_old_index_without_variant_fields_loads_uniform(tmp_path):
    cache_dir = _build(tmp_path)
    index = json.loads((cache_dir / "index.json").read_text())
    for item in index["items"]:
        item.pop("variant_kinds")
        item.pop("variant_weights")
    index.pop("variant_weight_overrides")
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")

    dataset = CachedDataset(cache_dir)
    samples = [dataset[0] for _ in range(20)]
    assert all(sample["variant_kind"] == "caption" for sample in samples)
    assert {int(sample["text"].shape[0]) for sample in samples} <= set(dataset.items[0]["text_lengths"])


def test_index_length_mismatch_is_rejected(tmp_path):
    cache_dir = _build(tmp_path)
    index = json.loads((cache_dir / "index.json").read_text())
    index["items"][0]["caption_sha256"] = index["items"][0]["caption_sha256"][:2]
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="caption_sha256"):
        CachedDataset(cache_dir)


def test_max_text_length_change_invalidates_embeddings(tmp_path):
    cache_dir = _build(tmp_path)
    manifest = tmp_path / "data" / "meta.jsonl"
    encoder = FakeTextEncoder()
    build_cache(
        manifest, cache_dir, text_encoder_factory=lambda: encoder,
        latent_encoder_factory=fake_latent_encoder_factory(8),
        target_area=256 * 256, max_text_length=4096, progress=False, device="cpu")
    assert encoder.calls and all(max_length == 4096 for _, max_length in encoder.calls)
    index = json.loads((cache_dir / "index.json").read_text())
    assert index["max_text_length"] == 4096
