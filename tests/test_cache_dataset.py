"""Cache and dataset tests: buckets, manifests, resume, item layout, batching."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
import torch
from safetensors.torch import safe_open

from moss_lora.cache import (
    DEFAULT_MAX_TEXT_LENGTH,
    assign_bucket,
    build_buckets,
    build_cache,
    caption_variants,
    collect_items,
)
from moss_lora.dataset import BucketBatchSampler, CachedDataset, _seed_worker, collate_batch
from moss_lora.utils import sha256_file, sha256_text
from tests.helpers import (
    FakeTextEncoder,
    fake_latent_encoder_factory,
    gradient_image,
)

# --------------------------------------------------------------------------- buckets


def test_build_buckets_invariants_1m():
    buckets = build_buckets(1024 * 1024)
    assert (1024, 1024) in buckets
    for width, height in buckets:
        assert width % 64 == 0 and height % 64 == 0
        assert 512 <= width <= 2048 and 512 <= height <= 2048
        area = width * height
        assert abs(area - 1024**2) / 1024**2 < 0.08
    for ratio_w, ratio_h in [(1, 1), (4, 3), (3, 4), (16, 9), (9, 16), (3, 2), (2, 3),
                             (5, 4), (4, 5), (2, 1), (1, 2)]:
        aspect = ratio_w / ratio_h
        best = min(buckets, key=lambda b: abs(math.log(b[0] / b[1]) - math.log(aspect)))
        assert abs(math.log(best[0] / best[1]) - math.log(aspect)) < 0.05, (ratio_w, ratio_h, best)


def test_build_buckets_small_target_preserves_aspect():
    buckets = build_buckets(512 * 512)
    wide = max(buckets, key=lambda b: b[0] / b[1])
    assert wide[0] / wide[1] >= 1.9  # 2:1 survived
    tall = min(buckets, key=lambda b: b[0] / b[1])
    assert tall[1] / tall[0] >= 1.9  # 1:2 survived
    for width, height in buckets:
        assert width % 64 == 0 and height % 64 == 0
        assert 256 <= width <= 2048 and 256 <= height <= 2048


def test_assign_bucket_nearest_aspect():
    buckets = [(1024, 1024), (1344, 768), (768, 1344)]
    assert assign_bucket(1024, 1024, buckets) == (1024, 1024)
    assert assign_bucket(1000, 560, buckets) == (1344, 768)
    assert assign_bucket(560, 1000, buckets) == (768, 1344)


# --------------------------------------------------------------------------- captions


def test_caption_variants_dict_long_medium_short():
    raw = {"caption": {"long": "very long", "medium": "", "short": "tiny"}}
    assert caption_variants(raw) == ["very long", "tiny"]
    # JSON string form behaves identically
    assert caption_variants(json.dumps(raw)) == ["very long", "tiny"]
    assert caption_variants({"caption": {"long": "a", "medium": "a"}}) == ["a"]


def test_caption_variants_layout_serialization():
    layout = {"summary": {"caption": "标题"}, "items": [{"text": "hello"}]}
    variants = caption_variants(layout)
    assert variants == [json.dumps(layout, ensure_ascii=False, separators=(",", ":"))]
    assert "标题" in variants[0]  # ensure_ascii=False


def test_caption_variants_strings_and_dicts():
    assert caption_variants("plain caption") == ["plain caption"]
    assert caption_variants({"text": "from text key"}) == ["from text key"]
    assert caption_variants({"prompt": "from prompt key"}) == ["from prompt key"]
    with pytest.raises(ValueError):
        caption_variants("   ")
    with pytest.raises(ValueError):
        caption_variants(42)


def test_caption_variants_image_description_schema():
    """Annotated/sampled sidecars use image_description, not caption/text/prompt."""
    raw = {"year": 2025, "preference_level": "best",
           "image_description": "@artist, landscape, sunset, city lights",
           "extra_tags": ["landscape", "sunset"]}
    assert caption_variants(raw) == ["@artist, landscape, sunset, city lights"]
    assert caption_variants(json.dumps(raw)) == ["@artist, landscape, sunset, city lights"]
    with pytest.raises(ValueError):
        caption_variants({**raw, "image_description": "   "})


# --------------------------------------------------------------------------- manifests


def make_image(path: Path, width: int, height: int, seed: int) -> Path:
    gradient_image(width, height, seed).save(path)
    return path


def test_collect_folder_manifests(tmp_path):
    make_image(tmp_path / "a.png", 64, 64, 0)
    (tmp_path / "a.txt").write_text("txt caption", encoding="utf-8")
    make_image(tmp_path / "b.png", 48, 48, 1)
    (tmp_path / "b.json").write_text('{"caption": {"long": "L", "short": "S"}}', encoding="utf-8")
    make_image(tmp_path / "c.png", 32, 32, 2)
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"file_name": "c.png", "caption": {"long": "meta long", "medium": "meta med"}}) + "\n",
        encoding="utf-8")
    make_image(tmp_path / "d.png", 32, 32, 3)
    (tmp_path / "metadata.csv").write_text("file_name,text\nd.png,csv caption\n", encoding="utf-8")
    make_image(tmp_path / "e.png", 32, 32, 4)  # no caption anywhere

    items = collect_items(tmp_path)
    raws = {path.name: raw for path, raw in items}
    assert raws["a.png"] == "txt caption"
    assert raws["b.png"] == '{"caption": {"long": "L", "short": "S"}}'
    assert raws["c.png"] == {"long": "meta long", "medium": "meta med"}
    assert raws["d.png"] == "csv caption"
    assert raws["e.png"] is None
    with pytest.raises(ValueError, match="Caption is empty"):
        caption_variants(raws["e.png"])


def test_collect_jsonl_keys(tmp_path):
    make_image(tmp_path / "x.png", 32, 32, 5)
    make_image(tmp_path / "y.png", 32, 32, 6)
    make_image(tmp_path / "z.png", 32, 32, 7)
    manifest = tmp_path / "data.jsonl"
    rows = [
        {"image": "x.png", "caption": "plain"},
        {"file_name": "y.png", "text": "text key"},
        {"file_name": "z.png", "layout": {"summary": {"caption": "lay"}, "items": []}},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    items = collect_items(manifest)
    assert items[0][1] == "plain"
    assert items[1][1] == "text key"
    assert caption_variants(items[2][1])[0].startswith('{"summary"')
    for missing in (["{}"], [json.dumps({"file_name": "nope.png", "text": "t"})]):
        bad = tmp_path / "bad.jsonl"
        bad.write_text("\n".join(missing), encoding="utf-8")
        with pytest.raises((ValueError, FileNotFoundError)):
            collect_items(bad)


def test_collect_jsonl_sidecar_captions(tmp_path):
    make_image(tmp_path / "a.png", 32, 32, 10)
    make_image(tmp_path / "b.png", 32, 32, 11)
    make_image(tmp_path / "c.png", 32, 32, 12)
    (tmp_path / "a.json").write_text('{"caption": {"long": "json sidecar"}}', encoding="utf-8")
    (tmp_path / "b.txt").write_text("txt sidecar\n", encoding="utf-8")
    manifest = tmp_path / "data.jsonl"
    rows = [{"image": "a.png"}, {"file_name": "b.png"}]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    items = collect_items(manifest)
    raws = {path.name: raw for path, raw in items}
    assert raws["a.png"] == '{"caption": {"long": "json sidecar"}}'
    assert raws["b.png"] == "txt sidecar\n"
    assert caption_variants(raws["a.png"]) == ["json sidecar"]

    missing = tmp_path / "missing.jsonl"
    missing.write_text(json.dumps({"file_name": "c.png"}) + "\n", encoding="utf-8")
    items = collect_items(missing)
    assert items[0][1] is None  # build_cache applies missing_caption (skip/error)
    with pytest.raises(ValueError, match="Caption is empty"):
        caption_variants(items[0][1])


# --------------------------------------------------------------------------- build + layout


def test_cache_build_files_and_index(tmp_path):
    cache_dir = tmp_path / "cache"
    root = tmp_path / "data"
    root.mkdir()
    sizes = [(256, 256), (256, 256), (320, 192), (192, 320)]
    rows = []
    for index, (width, height) in enumerate(sizes):
        make_image(root / f"img_{index}.png", width, height, index)
        rows.append({"image": f"img_{index}.png",
                     "caption": {"long": f"long {index}", "medium": f"med {index}",
                                 "short": f"short {index}"}})
    manifest = root / "meta.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    text_encoder = FakeTextEncoder(text_dim=32)
    latent_factory = fake_latent_encoder_factory(8)
    build_cache(manifest, cache_dir, text_encoder_factory=lambda: text_encoder,
                latent_encoder_factory=latent_factory, target_area=256 * 256,
                max_text_length=1024, revision="revX", negative_prompt="blurry", device="cpu",
                progress=False)

    # Item file naming: id "item_00000" maps to item_00000.safetensors (no double prefix).
    item = cache_dir / "item_00000.safetensors"
    assert item.is_file()
    assert not (cache_dir / "item_item_00000.safetensors").exists()
    with safe_open(str(item), framework="pt") as handle:
        keys = set(handle.keys())
        meta = handle.metadata()
        latent = handle.get_tensor("latent")
    assert "latent" in keys and "text_0" in keys and "text_1" in keys and "text_lengths" in keys
    assert latent.shape[0] == 8 and latent.shape[-1] == 16  # 256px -> 16 latent cells
    assert meta["revision"] == "revX"
    assert meta["image_sha256"] == sha256_file(root / "img_0.png")

    index = json.loads((cache_dir / "index.json").read_text())
    assert index["revision"] == "revX"
    assert index["latent_channels"] == 8 and index["text_dim"] == 32
    assert index["negative_prompt_sha256"] == sha256_text("blurry")
    first = index["items"][0]
    assert first["id"] == "item_00000"
    assert first["bucket_w"] == 256 and first["bucket_h"] == 256
    assert first["caption_sha256"] == [sha256_text(v) for v in ("long 0", "med 0", "short 0")]
    assert len(first["text_lengths"]) == 3 and all(length >= 3 for length in first["text_lengths"])

    negative = cache_dir / "negative.safetensors"
    with safe_open(str(negative), framework="pt") as handle:
        assert handle.get_tensor("embedding").dtype == torch.bfloat16
        assert int(handle.get_tensor("length")) == handle.get_tensor("embedding").shape[0]
    # Text embeddings stored bf16; encoder was unloaded after the text phase.
    with safe_open(str(item), framework="pt") as handle:
        assert handle.get_tensor("text_0").dtype == torch.bfloat16
    assert text_encoder.unloaded


def test_cache_resume_skips_factories_when_complete(tmp_path):
    cache_dir = tmp_path / "cache"
    root = tmp_path / "data"
    root.mkdir()
    make_image(root / "img.png", 256, 256, 9)
    manifest = root / "meta.jsonl"
    manifest.write_text(json.dumps({"image": "img.png", "caption": {"long": "L", "short": "S"}}) + "\n",
                        encoding="utf-8")
    kwargs = dict(target_area=256 * 256, max_text_length=1024, revision="revR",
                  negative_prompt="neg", device="cpu", progress=False)
    build_cache(manifest, cache_dir, text_encoder_factory=FakeTextEncoder,
                latent_encoder_factory=fake_latent_encoder_factory(8), **kwargs)

    def failing_text():
        raise AssertionError("text encoder must not load when the cache is complete")

    def failing_latent(device):
        raise AssertionError("VAE must not load when the cache is complete")

    build_cache(manifest, cache_dir, text_encoder_factory=failing_text,
                latent_encoder_factory=failing_latent, **kwargs)


def test_cache_resume_reuses_valid_text_file(tmp_path):
    cache_dir = tmp_path / "cache"
    root = tmp_path / "data"
    root.mkdir()
    make_image(root / "img.png", 256, 256, 10)
    manifest = root / "meta.jsonl"
    manifest.write_text(json.dumps({"image": "img.png", "caption": "hello"}) + "\n", encoding="utf-8")
    kwargs = dict(target_area=256 * 256, max_text_length=1024, revision=None,
                  negative_prompt="", device="cpu", progress=False)

    first = FakeTextEncoder()
    build_cache(manifest, cache_dir, text_encoder_factory=lambda: first,
                latent_encoder_factory=fake_latent_encoder_factory(8), **kwargs)
    second = FakeTextEncoder()
    build_cache(manifest, cache_dir, text_encoder_factory=lambda: second,
                latent_encoder_factory=fake_latent_encoder_factory(8), **kwargs)
    assert second.calls == []  # text reused from the valid cache
    assert (cache_dir / "item_00000.text.safetensors").exists() is False  # temp removed


def test_no_character_truncation_of_captions(tmp_path):
    cache_dir = tmp_path / "cache"
    root = tmp_path / "data"
    root.mkdir()
    make_image(root / "img.png", 256, 256, 11)
    long_caption = "word " * 500  # 2500 characters, far beyond max_text_length
    manifest = root / "meta.jsonl"
    manifest.write_text(json.dumps({"image": "img.png", "caption": long_caption}) + "\n", encoding="utf-8")
    encoder = FakeTextEncoder()
    build_cache(manifest, cache_dir,
                text_encoder_factory=lambda: encoder,
                latent_encoder_factory=fake_latent_encoder_factory(8),
                target_area=256 * 256, max_text_length=DEFAULT_MAX_TEXT_LENGTH,
                negative_prompt="", device="cpu", progress=False)
    prompt, max_length = encoder.calls[0]
    assert prompt == long_caption.strip()  # not cut at 1024 characters (caption_variants strips)
    assert max_length == DEFAULT_MAX_TEXT_LENGTH  # token limit passed to the tokenizer


def test_changed_caption_invalidates_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    root = tmp_path / "data"
    root.mkdir()
    make_image(root / "img.png", 256, 256, 12)
    manifest = root / "meta.jsonl"
    manifest.write_text(json.dumps({"image": "img.png", "caption": "first"}) + "\n", encoding="utf-8")
    kwargs = dict(target_area=256 * 256, max_text_length=1024, revision=None,
                  negative_prompt="", device="cpu", progress=False)
    build_cache(manifest, cache_dir, text_encoder_factory=FakeTextEncoder,
                latent_encoder_factory=fake_latent_encoder_factory(8), **kwargs)
    manifest.write_text(json.dumps({"image": "img.png", "caption": "second"}) + "\n", encoding="utf-8")
    encoder = FakeTextEncoder()
    build_cache(manifest, cache_dir, text_encoder_factory=lambda: encoder,
                latent_encoder_factory=fake_latent_encoder_factory(8), **kwargs)
    assert encoder.calls and encoder.calls[0][0] == "second"
    with safe_open(str(cache_dir / "item_00000.safetensors"), framework="pt") as handle:
        assert json.loads(handle.metadata()["caption_sha256"]) == [sha256_text("second")]
        assert torch.equal(handle.get_tensor("text_0"), encoder.encode("second"))


def test_cache_jsonl_missing_caption_uses_missing_caption_mode(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    make_image(root / "good.png", 256, 256, 13)
    make_image(root / "bad.png", 256, 256, 14)
    manifest = root / "meta.jsonl"
    manifest.write_text(
        json.dumps({"image": "good.png", "caption": "ok"}) + "\n"
        + json.dumps({"image": "bad.png"}) + "\n", encoding="utf-8")
    kwargs = dict(target_area=256 * 256, max_text_length=1024, revision=None,
                  negative_prompt="", device="cpu", progress=False)

    build_cache(manifest, tmp_path / "cache", text_encoder_factory=FakeTextEncoder,
                latent_encoder_factory=fake_latent_encoder_factory(8), **kwargs)
    index = json.loads((tmp_path / "cache" / "index.json").read_text())
    assert [Path(item["image"]).name for item in index["items"]] == ["good.png"]
    with pytest.raises(ValueError, match="bad.png"):
        build_cache(manifest, tmp_path / "cache_error", missing_caption="error", **kwargs)


def test_cache_skips_missing_and_empty_captions(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    root = tmp_path / "data"
    root.mkdir()
    make_image(root / "good.png", 256, 256, 20)
    (root / "good.txt").write_text("a good caption\n", encoding="utf-8")
    make_image(root / "missing.png", 256, 256, 21)  # no same-stem sidecar
    make_image(root / "empty.png", 256, 256, 22)
    (root / "empty.txt").write_text("  \n\t\n", encoding="utf-8")  # whitespace only
    kwargs = dict(target_area=256 * 256, max_text_length=1024, revision=None,
                  negative_prompt="", device="cpu", progress=False)

    build_cache(root, cache_dir, text_encoder_factory=FakeTextEncoder,
                latent_encoder_factory=fake_latent_encoder_factory(8), **kwargs)

    index = json.loads((cache_dir / "index.json").read_text())
    assert len(index["items"]) == 1
    assert index["items"][0]["id"] == "item_00000"
    assert index["items"][0]["image"].endswith("good.png")
    assert (cache_dir / "item_00000.safetensors").is_file()
    assert not (cache_dir / "item_00001.safetensors").exists()
    assert not (cache_dir / "item_00001.text.safetensors").exists()
    stderr = capsys.readouterr().err
    assert "missing.png" in stderr and "empty.png" in stderr
    assert "3 images, 2 skipped (no/empty caption)" in stderr

    # Re-running the same sorted input reproduces the same kept ids and skips,
    # so a complete cache loads no encoders.
    def fail_text():
        raise AssertionError("text encoder must not load when the kept cache is complete")

    def fail_latent(device):
        raise AssertionError("VAE must not load when the kept cache is complete")

    build_cache(root, cache_dir, text_encoder_factory=fail_text,
                latent_encoder_factory=fail_latent, **kwargs)
    assert json.loads((cache_dir / "index.json").read_text())["items"] == index["items"]


def test_cache_missing_caption_error_mode_reports_path(tmp_path):
    cache_dir = tmp_path / "cache"
    root = tmp_path / "data"
    root.mkdir()
    make_image(root / "good.png", 256, 256, 30)
    (root / "good.txt").write_text("a good caption\n", encoding="utf-8")
    make_image(root / "missing.png", 256, 256, 31)
    with pytest.raises(ValueError) as info:
        build_cache(root, cache_dir, text_encoder_factory=FakeTextEncoder,
                    latent_encoder_factory=fake_latent_encoder_factory(8),
                    target_area=256 * 256, max_text_length=1024, revision=None,
                    negative_prompt="", device="cpu", progress=False,
                    missing_caption="error")
    message = str(info.value)
    assert str(root / "missing.png") in message
    assert "--missing-caption skip" in message


def test_cache_all_captions_skipped_raises(tmp_path):
    cache_dir = tmp_path / "cache"
    root = tmp_path / "data"
    root.mkdir()
    make_image(root / "a.png", 256, 256, 40)
    make_image(root / "b.png", 256, 256, 41)
    with pytest.raises(ValueError, match="No usable captions"):
        build_cache(root, cache_dir, target_area=256 * 256, max_text_length=1024,
                    revision=None, negative_prompt="", device="cpu", progress=False)


# --------------------------------------------------------------------------- dataset


def test_dataset_and_sampler(tiny_cache):
    dataset = CachedDataset(tiny_cache)
    assert len(dataset) == 4
    buckets = [(item["bucket_h"], item["bucket_w"]) for item in dataset.items]
    sampler = BucketBatchSampler(buckets, batch_size=2, drop_last=False, seed=42)
    assert set(sampler.groups) == {(256, 256), (192, 320), (320, 192)}
    batches = sampler.batches_for_epoch(0)
    for batch in batches:
        shapes = {buckets[i] for i in batch}
        assert len(shapes) == 1  # exact bucket grouping
    again = sampler.batches_for_epoch(0)
    assert again == batches  # deterministic per epoch
    assert sampler.batches_for_epoch(1) != batches or len(batches) == 1

    sample = dataset[0]
    assert sample["latents"].shape[0] == 8
    assert sample["text"].dtype == torch.bfloat16

    drop = BucketBatchSampler(buckets, batch_size=2, drop_last=True, seed=0)
    assert all(len(batch) == 2 for batch in drop.batches_for_epoch(0))


def test_collate_pads_and_masks():
    batch = [
        {"latents": torch.zeros(8, 4, 4), "text": torch.ones(3, 5).to(torch.bfloat16), "id": "a"},
        {"latents": torch.zeros(8, 4, 4), "text": torch.ones(6, 5).to(torch.bfloat16), "id": "b"},
    ]
    collated = collate_batch(batch)
    assert collated["latents"].shape == (2, 8, 4, 4)
    assert collated["text"].shape == (2, 6, 5)
    assert collated["mask"].tolist() == [[True, True, True, False, False, False],
                                         [True] * 6]
    assert collated["ids"] == ["a", "b"]
    with pytest.raises(RuntimeError, match="mixes latent shapes"):
        collate_batch(batch + [dict(batch[0], latents=torch.zeros(8, 5, 5))])


def test_seed_worker_seeds_python_random():
    torch.manual_seed(1234)
    random.seed(0)
    _seed_worker(0)
    first = random.random()
    torch.manual_seed(1234)
    random.seed(0)
    _seed_worker(0)
    assert random.random() == first
