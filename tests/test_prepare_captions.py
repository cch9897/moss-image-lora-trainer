"""Caption preprocessing tests: parsing, variants, filters, reports, cache hand-off."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import pytest

from moss_lora.cache import build_cache, caption_variants, collect_items
from moss_lora.dataset import CachedDataset
from scripts.prepare_captions import (
    PrepareConfig,
    PreparedItem,
    PrepareError,
    SidecarError,
    build_captions,
    crop_fraction,
    dedupe_tags,
    estimate_tokens,
    load_tokenizer,
    normalize_name,
    oriented_size,
    parse_json_sidecar,
    parse_raw_prompt,
    prepare_dataset,
    token_length,
)
from tests.helpers import FakeTextEncoder, fake_latent_encoder_factory, gradient_image

TEXT_EN = "A calm city street at dusk, with clear sign lettering and a raised hand."
TEXT_ZH = "这是一幅安静的城市街景，画面中有招牌文字和一只抬起的手。"
RAW = f"""@sample_handle, artist:alice, character:hero, landscape, sunset, city lights, sign

{TEXT_ZH}

{TEXT_EN}
#vlm"""


def write_image(path: Path, width: int = 256, height: int = 256, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gradient_image(width, height, seed).save(path)
    return path


def write_case(root: Path, name: str, text: str, *, width: int = 256, height: int = 256,
               seed: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image = write_image(root / f"{name}.png", width, height, seed)
    image.with_suffix(".txt").write_text(text, encoding="utf-8")
    return image


def config(**overrides) -> PrepareConfig:
    values = dict(tokenizer="none", target_area=256 * 256)
    values.update(overrides)
    return PrepareConfig(**values)


def item_from_raw(raw: str = RAW, **overrides) -> PreparedItem:
    parsed = parse_raw_prompt(raw)
    tags, removed = dedupe_tags(parsed.tags)
    values = dict(
        image=Path("/tmp/image.png"),
        source_rel="image.png",
        parsed=parsed,
        tags=tags,
        removed_negatives=removed,
        artist=parsed.artist_values[0] if parsed.artist_values else "unknown",
        artist_missing=not parsed.artist_values,
        characters=parsed.characters,
        source_handle=parsed.source_handle,
        prose_en=parsed.prose_en,
        prose_zh=parsed.prose_zh,
        preference_level=parsed.preference_level,
        layout=parsed.layout,
        artist_values=parsed.artist_values,
    )
    values.update(overrides)
    return PreparedItem(**values)


# --------------------------------------------------------------------------- raw parsing


def test_parse_raw_prompt_tags_prose_and_vlm():
    parsed = parse_raw_prompt(RAW)
    assert parsed.tag_line_found is True
    assert parsed.source_handle == "@sample_handle"
    assert parsed.artist_values == ["alice"]
    assert parsed.characters == ["hero"]
    assert parsed.tags == ["landscape", "sunset", "city lights", "sign"]
    assert parsed.prose_zh == [TEXT_ZH]
    assert parsed.prose_en == [TEXT_EN]
    assert "#vlm" not in " ".join(parsed.tags + parsed.prose_en + parsed.prose_zh)


def test_fullwidth_handle_and_name_normalization():
    raw = "＠sample, artist:  Alice   Smith , character:hero, character:hero, landscape"
    parsed = parse_raw_prompt(raw)
    assert parsed.source_handle == "@sample"
    assert parsed.artist_values == ["Alice Smith"]
    assert parsed.characters == ["hero"]
    assert parsed.tags == ["landscape"]
    assert normalize_name("  ＠Foo   Bar  ") == "@Foo Bar"


def test_tag_line_only_prompt_is_usable():
    parsed = parse_raw_prompt("@sample, artist:alice, landscape, sunset")
    assert parsed.prose_en == [] and parsed.prose_zh == []
    assert parsed.tags == ["landscape", "sunset"]


def test_empty_prompt_rejected():
    with pytest.raises(SidecarError):
        parse_raw_prompt("   \n#vlm\n")


# --------------------------------------------------------------------------- negatives


def test_negative_tags_removed_from_positive_captions():
    raw = (
        "@sample, artist:alice, landscape, sunset, worst_quality, bad anatomy, watermark, "
        "JPEG artifacts, bad hands, extra fingers, missing fingers, city lights"
    )
    parsed = parse_raw_prompt(raw)
    tags, removed = dedupe_tags(parsed.tags)
    assert tags == ["landscape", "sunset", "city lights"]
    assert removed["worst quality"] == 1
    assert removed["bad anatomy"] == 1
    assert removed["watermark"] == 1
    assert removed["jpeg artifacts"] == 1
    assert removed["bad hands"] == 1
    assert removed["extra fingers"] == 1
    assert removed["missing fingers"] == 1

    captions = build_captions(item_from_raw(raw), trigger="trigger", language="en", max_tags=40)
    assert "watermark" not in captions["long"]
    assert "worst quality" not in captions["short"]


# --------------------------------------------------------------------------- caption variants


def test_long_medium_short_include_artist_and_character():
    tags = ", ".join(f"tag{index}" for index in range(30))
    raw = (
        f"@sample, artist:alice, character:hero, character:sidekick, {tags}"
        f"\n\n{TEXT_EN}\n\nSecond paragraph."
    )
    item = item_from_raw(raw)
    captions = build_captions(item, trigger="mystyle", language="en", max_tags=40)

    for name in ("long", "medium", "short"):
        assert captions[name].startswith(
            "mystyle, artist:alice, character:hero, character:sidekick"
        )
    assert captions["long"].count("tag") == 30
    assert captions["long"].endswith("Second paragraph.")
    assert captions["medium"].count("tag") == 25
    assert captions["medium"].endswith(TEXT_EN)
    assert "Second paragraph" not in captions["medium"]
    assert captions["short"].count("tag") == 12
    assert TEXT_EN not in captions["short"]
    assert "artist:alice" not in captions["no_artist"]
    assert captions["no_artist"].startswith("mystyle, character:hero, character:sidekick")


def test_max_tags_caps_long_and_variants():
    raw = f"@sample, artist:alice, {', '.join(f'tag{i}' for i in range(30))}\n\n{TEXT_EN}"
    captions = build_captions(item_from_raw(raw), trigger="t", language="en", max_tags=5)
    assert captions["long"].count("tag") == 5
    assert captions["medium"].count("tag") == 5
    assert captions["short"].count("tag") == 5


def test_exactly_one_prose_language_selected():
    item = item_from_raw()
    english = build_captions(item, trigger="t", language="en", max_tags=40)
    chinese = build_captions(item, trigger="t", language="zh", max_tags=40)
    automatic = build_captions(item, trigger="t", language="auto", max_tags=40)
    assert english["language_used"] == "en"
    assert chinese["language_used"] == "zh"
    assert automatic["language_used"] == "zh"
    assert TEXT_EN in english["long"] and TEXT_ZH not in english["long"]
    assert TEXT_ZH in chinese["long"] and TEXT_EN not in chinese["long"]
    assert TEXT_ZH in automatic["long"] and TEXT_EN not in automatic["long"]


def test_requested_language_without_prose_never_crosses_over():
    item = item_from_raw(f"@sample, artist:alice, landscape\n\n{TEXT_ZH}")
    english = build_captions(item, trigger="t", language="en", max_tags=40)
    assert english["language_used"] == "none"
    assert TEXT_ZH not in english["long"]
    assert english["language_fallback"] is True


# --------------------------------------------------------------------------- JSON sidecars


def test_json_sidecar_fields_and_layout():
    layout = {"summary": {"caption": "poster"}, "items": [{"label": "text", "text": "HELLO"}]}
    parsed = parse_json_sidecar({
        "artist": "alice",
        "character": ["hero", "hero", "sidekick"],
        "extra_tags": "landscape, sunset, watermark, sign",
        "image_description": TEXT_EN,
        "preference_level": "best",
        "layout": layout,
    })
    assert parsed.artist_values == ["alice"]
    assert parsed.characters == ["hero", "sidekick"]
    assert parsed.tags == ["landscape", "sunset", "watermark", "sign"]
    assert parsed.prose_en == [TEXT_EN]
    assert parsed.preference_level == "best"
    assert parsed.layout == layout

    item = item_from_raw(RAW, parsed=parsed, tags=dedupe_tags(parsed.tags)[0],
                         artist="alice", characters=parsed.characters,
                         prose_en=parsed.prose_en, prose_zh=parsed.prose_zh,
                         preference_level="best", layout=layout)
    captions = build_captions(item, trigger="t", language="en", max_tags=40)
    assert "artist:alice" in captions["long"]
    assert "watermark" not in captions["long"]
    assert "sign" in captions["long"]


def test_json_sidecar_caption_long_and_nested_layout():
    parsed = parse_json_sidecar({"caption": {"long": TEXT_EN, "short": "short"}})
    assert parsed.prose_en == [TEXT_EN]
    assert parsed.prose_zh == []

    nested = parse_json_sidecar({"caption": json.dumps({"artist": "bob", "extra_tags": ["sunset"],
                                                       "image_description": TEXT_EN})})
    assert nested.artist_values == ["bob"]
    assert nested.tags == ["sunset"]
    assert nested.prose_en == [TEXT_EN]

    layout_only = parse_json_sidecar({"layout": {"summary": {"caption": TEXT_EN}, "items": []}})
    assert layout_only.prose_en == [TEXT_EN]


def test_json_sidecar_without_usable_fields_rejected():
    with pytest.raises(SidecarError):
        parse_json_sidecar({"preference_level": "best"})


# --------------------------------------------------------------------------- files/reports


def test_missing_and_unparsable_sidecars_are_reported(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    write_image(root / "good.png", seed=1)
    (root / "good.txt").write_text(RAW, encoding="utf-8")
    write_image(root / "missing.png", seed=2)
    write_image(root / "empty.png", seed=3)
    (root / "empty.txt").write_text("  \n#vlm\n", encoding="utf-8")

    output = tmp_path / "out"
    report = prepare_dataset(root, output, config())
    assert report["images"]["written"] == 1
    assert report["images"]["missing_sidecar"] == 1
    assert report["images"]["unparsable_sidecar"] == 1
    assert report["images"]["supported"] == 3
    assert (output / "metadata.jsonl").is_file()
    assert (output / "report.md").is_file()
    parsed_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert parsed_report["images"]["written"] == 1
    assert {entry["reason"] for entry in parsed_report["skipped"]} == {
        "missing_sidecar", "unparsable_sidecar"
    }

    rows = [json.loads(line) for line in (output / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["file_name"] == "images/000001.png"
    assert row["artist"] == "alice"
    assert row["character"] == ["hero"]
    assert row["source_handle"] == "@sample_handle"
    assert row["preference_level"] is None
    assert row["tags"][:2] == ["landscape", "sunset"]
    assert row["caption"]["long"].startswith("mystyle, artist:alice, character:hero")
    assert "artist:alice" not in row["caption_no_artist"]


def test_empty_json_falls_back_to_txt_and_keeps_manifest_metadata(tmp_path):
    root = tmp_path / "data"
    write_image(root / "a.png", seed=1)
    (root / "a.json").write_text("", encoding="utf-8")
    (root / "a.txt").write_text(RAW, encoding="utf-8")
    (root / "metadata.jsonl").write_text(
        json.dumps({"file_name": "a.png", "preference_level": "best"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    report = prepare_dataset(root, output, config())
    assert report["images"]["written"] == 1
    assert report["sidecars"]["both_json_and_txt"] == 1
    row = json.loads((output / "metadata.jsonl").read_text(encoding="utf-8"))
    assert row["preference_level"] == "best"
    assert row["artist"] == "alice"


def test_output_metadata_is_consumable_by_cache(tmp_path):
    root = tmp_path / "data"
    write_case(root, "a", RAW, seed=1)
    write_case(root, "b", "@h, artist:bob, character:sidekick, landscape, sunset, sign\n\n"
                            "Another English description with a sign.", seed=2)
    output = tmp_path / "processed"
    report = prepare_dataset(root, output, config())
    assert report["images"]["written"] == 2

    items = collect_items(output)
    assert len(items) == 2
    for _, raw in items:
        variants = caption_variants(raw)
        # The cache dedupes identical variants (a one-paragraph long caption can
        # equal medium); metadata still carries all three keys.
        assert 2 <= len(variants) <= 3
        assert all("artist:" in variant for variant in variants)

    cache = tmp_path / "cache"
    encoder = FakeTextEncoder(text_dim=32)
    build_cache(
        output,
        cache,
        text_encoder_factory=lambda: encoder,
        latent_encoder_factory=fake_latent_encoder_factory(8),
        target_area=256 * 256,
        max_text_length=1024,
        revision="test",
        negative_prompt="negative",
        device="cpu",
        progress=False,
    )
    dataset = CachedDataset(cache)
    assert len(dataset) == 2
    random.seed(0)
    sample = dataset[0]
    assert sample["latents"].shape == (8, 16, 16)
    assert sample["text"].ndim == 2
    assert sample["bucket"] == (256, 256)


def test_symlink_copy_and_symlink_fallback(tmp_path, monkeypatch):
    root = tmp_path / "data"
    source = write_case(root, "a", RAW, seed=1)

    linked = tmp_path / "linked"
    prepare_dataset(root, linked, config())
    link = linked / "images" / "000001.png"
    assert link.is_symlink()
    assert not os.readlink(link).startswith("/")
    assert link.resolve() == source.resolve()

    copied = tmp_path / "copied"
    prepare_dataset(root, copied, config(copy_images=True))
    copy = copied / "images" / "000001.png"
    assert not copy.is_symlink()
    assert copy.read_bytes() == source.read_bytes()

    fallback = tmp_path / "fallback"
    real_symlink = os.symlink

    def broken_symlink(*args, **kwargs):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(os, "symlink", broken_symlink)
    prepare_dataset(root, fallback, config())
    monkeypatch.setattr(os, "symlink", real_symlink)
    fallback_copy = fallback / "images" / "000001.png"
    assert not fallback_copy.is_symlink()
    assert fallback_copy.read_bytes() == source.read_bytes()


# --------------------------------------------------------------------------- filters


def test_artist_count_and_per_artist_filters(tmp_path):
    root = tmp_path / "data"
    for index in range(3):
        write_case(root, f"a{index}",
                   f"@h, artist:alice, landscape, sunset\n\nTEXT English {index}.", seed=index)
    write_case(root, "b0", "@h, artist:bob, landscape, sunset\n\nTEXT English B.", seed=10)

    filtered = tmp_path / "filtered"
    report = prepare_dataset(root, filtered, config(min_artist_count=2))
    assert report["images"]["written"] == 3
    assert report["images"]["filtered"]["min_artist_count"] == 1
    assert report["artists"]["long_tail"]["count_1"] == 1

    capped = tmp_path / "capped"
    report = prepare_dataset(root, capped, config(max_per_artist=2))
    assert report["images"]["written"] == 3
    assert report["images"]["filtered"]["max_per_artist"] == 1

    required = tmp_path / "required"
    with pytest.raises(PrepareError):
        prepare_dataset(root, required, config(require_tags=("missing",)))
    report = json.loads((required / "report.json").read_text(encoding="utf-8"))
    assert report["images"]["written"] == 0
    assert report["images"]["filtered"]["require_tag"] == 4


def test_artist_map_and_trigger(tmp_path):
    root = tmp_path / "data"
    write_case(root, "a", RAW, seed=1)
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"alice": "Alice Canonical"}), encoding="utf-8")
    from scripts.prepare_captions import load_artist_map

    output = tmp_path / "out"
    report = prepare_dataset(
        root, output,
        config(artist_map=load_artist_map(mapping), trigger="mystyle"),
    )
    row = json.loads((output / "metadata.jsonl").read_text(encoding="utf-8"))
    assert row["artist"] == "Alice Canonical"
    assert row["caption"]["long"].startswith("mystyle, artist:Alice Canonical")
    assert report["artists"]["artist_map_entries"] == 1


def test_jsonl_input_is_sorted_and_deterministic(tmp_path):
    root = tmp_path / "data"
    write_image(root / "b.png", seed=2)
    write_image(root / "a.png", seed=1)
    manifest = root / "data.jsonl"
    rows = [
        {"image": "b.png", "artist": "bob", "extra_tags": ["sunset"], "image_description": TEXT_EN},
        {"file_name": "a.png", "artist": "alice", "extra_tags": ["sunset"], "image_description": TEXT_EN},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "out"
    prepare_dataset(manifest, output, config())
    first = (output / "metadata.jsonl").read_text(encoding="utf-8")
    files = [json.loads(line)["file_name"] for line in first.splitlines()]
    assert files == ["images/000001.png", "images/000002.png"]

    prepare_dataset(manifest, output, config(force=True))
    assert (output / "metadata.jsonl").read_text(encoding="utf-8") == first


def test_duplicate_manifest_rows_reported(tmp_path):
    root = tmp_path / "data"
    write_image(root / "a.png", seed=1)
    manifest = root / "data.jsonl"
    row = {"image": "a.png", "caption": RAW}
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    report = prepare_dataset(manifest, tmp_path / "out", config())
    assert report["images"]["written"] == 1
    assert report["sidecars"]["duplicate_manifest_rows"] == 1


def test_jsonl_missing_image_reported(tmp_path):
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(json.dumps({"image": "gone.png", "caption": "text"}) + "\n", encoding="utf-8")
    with pytest.raises(PrepareError):
        prepare_dataset(manifest, tmp_path / "out", config())


# --------------------------------------------------------------------------- tokenizer/crop


def test_tokenizer_fallback_and_token_lengths(tmp_path):
    tokenizer, path, error = load_tokenizer("/definitely/not/a/tokenizer")
    assert tokenizer is None
    assert path is None
    assert error
    assert token_length(None, "hello world") == estimate_tokens("hello world") >= 2

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return list(range(len(text.split())))

    assert token_length(FakeTokenizer(), "one two three") == 3

    root = tmp_path / "data"
    write_case(root, "a", RAW, seed=1)
    report = prepare_dataset(root, tmp_path / "out", config(tokenizer="none"))
    assert report["tokens"]["mode"] == "estimate"
    assert report["tokens"]["long"]["p50"] > 0
    assert report["tokens"]["long"]["over_2048"]["pct"] == 0.0


def test_crop_risk_math_and_report(tmp_path):
    assert crop_fraction(1000, 1000, (1024, 1024)) == 0.0
    # scale = max(1024/1000, 1024/2000) = 1.024 -> 1024x2048 then center-crop.
    expected = 1.0 - (1024 * 1024) / (1024 * 2048)
    assert crop_fraction(1000, 2000, (1024, 1024)) == pytest.approx(expected)
    assert oriented_size(4000, 3000, 6) == (3000, 4000)
    assert oriented_size(4000, 3000, 1) == (4000, 3000)

    root = tmp_path / "data"
    write_case(root, "tall", "@h, artist:alice, landscape\n\n" + TEXT_EN,
               width=256, height=1024, seed=1)
    write_case(root, "square", "@h, artist:alice, landscape\n\n" + TEXT_EN,
               width=256, height=256, seed=2)
    report = prepare_dataset(root, tmp_path / "out", config())
    assert report["crop_risk"]["items"] == 2
    assert report["crop_risk"]["over_5pct"] == 1
    assert report["crop_risk"]["worst"][0]["source"] == "tall.png"
    assert report["crop_risk"]["worst"][0]["crop_pct"] > 5
    assert math.isclose(report["crop_risk"]["max_crop_pct"],
                        report["crop_risk"]["worst"][0]["crop_pct"], abs_tol=0.01)


def test_text_hand_coverage_and_duplicate_hashes(tmp_path):
    root = tmp_path / "data"
    write_case(root, "a", "@h, artist:alice, landscape, sign, holding\n\n" + TEXT_EN, seed=1)
    write_case(root, "b", "@h, artist:alice, landscape, sign, holding\n\n" + TEXT_EN, seed=1)
    report = prepare_dataset(root, tmp_path / "out", config())
    tags = report["text_hand_coverage"]["tags"]["terms"]
    assert tags["sign"]["count"] >= 1
    assert tags["holding"]["count"] >= 1
    assert report["duplicates"]["images"]["duplicate_groups"] == 1
    assert report["duplicates"]["captions"]["duplicate_groups"] == 1


# --------------------------------------------------------------------------- CLI


def test_cli_writes_metadata(tmp_path):
    from scripts.prepare_captions import main

    root = tmp_path / "data"
    write_case(root, "a", RAW, seed=1)
    output = tmp_path / "out"
    assert main([
        "--input", str(root), "--output", str(output), "--language", "en",
        "--trigger", "mystyle", "--tokenizer", "none", "--target-area", "65536",
    ]) == 0
    assert (output / "metadata.jsonl").is_file()
    assert (output / "report.json").is_file()


def test_empty_output_rejected_until_force(tmp_path):
    root = tmp_path / "data"
    write_case(root, "a", RAW, seed=1)
    output = tmp_path / "out"
    prepare_dataset(root, output, config())
    with pytest.raises(PrepareError):
        prepare_dataset(root, output, config())
    prepare_dataset(root, output, config(force=True))
