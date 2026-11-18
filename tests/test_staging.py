"""Staging tests: filtering, artists, exact/near dedupe, index and reports."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moss_lora.staging import (
    DedupeIndex,
    HashRecord,
    ImageInfo,
    RejectedImage,
    StagingConfig,
    classify_suffix,
    inspect_image,
    load_artist_map,
    load_current_records,
    phash64,
    phash_hamming,
    resolve_artist,
    stage_pool,
)
from tests.helpers import gradient_image


def save(path: Path, image: Image.Image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def noise_image(seed: int, size: int = 64) -> Image.Image:
    """Structurally unrelated RGB noise (gradient fixtures share a pHash)."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8), mode="RGB")


def config(tmp_path: Path) -> StagingConfig:
    return StagingConfig(
        input_root=tmp_path / "pool", output_dir=tmp_path / "staged",
        min_side=16, max_side=1000, max_megapixels=1.0,
        min_aspect=0.2, max_aspect=5.0, link_mode="symlink")


def test_classify_suffix(tmp_path):
    assert classify_suffix(Path("a.JPG")) == "image"
    assert classify_suffix(Path("a.gif")) == "gif"
    assert classify_suffix(Path("a.mp4")) == "video"
    assert classify_suffix(Path("a.tiff")) == "unsupported"
    assert classify_suffix(Path("a.json")) == "sidecar"
    assert classify_suffix(Path("a.psd")) == "other"


def test_artist_resolution_and_map(tmp_path):
    resolution = resolve_artist("alice + bob/image.png", known_artists=["alice", "bob"])
    assert resolution.artists == ("alice", "bob") and resolution.artist == "alice"
    ambiguous = resolve_artist("alice + carol/image.png", known_artists=["alice"])
    assert ambiguous.artists == ("alice + carol",) and ambiguous.ambiguous
    unknown = resolve_artist("misc/image.png")
    assert unknown.artist == "unknown" and unknown.missing and not unknown.artists
    nested = resolve_artist("artist_a/sub/deep/image.png",
                            artist_map={"artist a": "Artist A Canonical"})
    assert nested.artist == "Artist A Canonical"
    mapping_path = tmp_path / "artist_map.json"
    mapping_path.write_text(json.dumps({"Alias_One": "Canonical"}), encoding="utf-8")
    mapping = load_artist_map(mapping_path)
    assert resolve_artist("alias one/x.png", artist_map=mapping).artist == "Canonical"


def test_inspect_image_gates_and_alpha(tmp_path):
    (tmp_path / "pool").mkdir()
    cfg = config(tmp_path)
    good = save(tmp_path / "pool/artist/good.png", gradient_image(64, 64, 1))
    info = inspect_image(good, cfg)
    assert (info.width, info.height) == (64, 64)
    assert len(info.sha256) == 64 and len(info.phash) == 16
    assert not info.alpha_composited

    rgba = gradient_image(64, 64, 16).convert("RGBA")
    rgba.putalpha(128)
    rgba_path = save(tmp_path / "pool/artist/alpha.png", rgba)
    info = inspect_image(rgba_path, cfg)
    assert info.alpha_composited

    with pytest.raises(RejectedImage) as tiny:
        inspect_image(save(tmp_path / "pool/artist/tiny.png", gradient_image(8, 16, 2)), cfg)
    assert getattr(tiny.value, "reason") == "too_small"
    blank = save(tmp_path / "pool/artist/blank.png", Image.new("RGB", (64, 64), (20, 20, 20)))
    with pytest.raises(RejectedImage) as blank_error:
        inspect_image(blank, cfg)
    assert getattr(blank_error.value, "reason") == "blank"
    wide = save(tmp_path / "pool/artist/wide.png", gradient_image(200, 20, 3))
    with pytest.raises(RejectedImage) as aspect:
        inspect_image(wide, cfg)
    assert getattr(aspect.value, "reason") == "bad_aspect"
    broken = tmp_path / "pool/artist/broken.jpg"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"not an image")
    with pytest.raises(RejectedImage) as unreadable:
        inspect_image(broken, cfg)
    assert getattr(unreadable.value, "reason") == "unreadable"


def test_phash_hamming():
    image = gradient_image(64, 64, 4)
    assert phash64(image) == phash64(image)
    assert phash_hamming(phash64(image), phash64(image)) == 0
    other = gradient_image(64, 64, 5)
    assert 0 <= phash_hamming(phash64(image), phash64(other)) <= 64


def test_dedupe_index_exact_near_and_embedding():
    base = ImageInfo(64, 64, "a" * 64, "0000000000000000")
    index = DedupeIndex()
    index.add(base, "base.png", embedding=[1.0, 0.0])
    exact = index.check(ImageInfo(64, 64, "a" * 64, "1111111111111111"), "exact.png")
    assert exact.status == "exact_dup" and exact.nearest == "base.png"
    near = index.check(ImageInfo(64, 64, "b" * 64, "0000000000000001"), "near.png")
    assert near.status == "near_dup" and near.reason == "phash"
    other = index.check(ImageInfo(64, 64, "c" * 64, "ffffffffffffffff"),
                        "other.png", embedding=[0.0, 1.0])
    assert other.status == "new"
    embedding_near = index.check(ImageInfo(64, 64, "d" * 64, "ffffffffffffffff"),
                                 "embed.png", embedding=[1.0, 0.01])
    assert embedding_near.status == "near_dup" and embedding_near.reason == "embedding"


def make_pool(tmp_path: Path) -> tuple[Path, dict]:
    pool = tmp_path / "pool"
    good = save(pool / "artist_a/dup.png", gradient_image(64, 64, 10))
    save(pool / "artist_a/good.png", Image.open(good).copy())
    near_image = Image.open(good).copy()
    near_image.putpixel((0, 0), (255, 255, 255))
    save(pool / "artist_a/near.png", near_image)
    save(pool / "misc/unknown.png", noise_image(11))
    save(pool / "alice + bob/multi.png", noise_image(12))
    save(pool / "bad/tiny.png", gradient_image(8, 16, 13))
    save(pool / "bad/blank.png", Image.new("RGB", (64, 64), (20, 20, 20)))
    save(pool / "bad/wide.png", gradient_image(200, 20, 14))
    (pool / "bad").mkdir(exist_ok=True)
    (pool / "bad/broken.jpg").write_bytes(b"not an image")
    (pool / "bad/anim.gif").write_bytes(b"GIF89a")
    (pool / "bad/clip.mp4").write_bytes(b"\x00")
    (pool / "bad/photo.tif").write_bytes(b"II*\x00")
    save(pool / ".hidden/ignored.png", gradient_image(64, 64, 15))
    return pool, {}


def test_stage_pool_reports_index_and_links(tmp_path):
    pool, _ = make_pool(tmp_path)
    cfg = config(tmp_path)
    report = stage_pool(cfg, artist_map={"artist a": "Artist A"},
                        known_artists={"alice", "bob"}, progress=False)

    assert report["accepted"] == 3
    assert report["counts"]["exact_dup"] == 1
    assert report["counts"]["near_dup"] == 1
    assert report["counts"]["gif"] == 1
    assert report["counts"]["video"] == 1
    assert report["counts"]["unsupported"] == 1
    assert report["counts"]["too_small"] == 1
    assert report["counts"]["blank"] == 1
    assert report["counts"]["bad_aspect"] == 1
    assert report["counts"]["unreadable"] == 1
    assert report["exact_duplicate_groups"][0]["sha256"]
    assert len(report["exact_duplicate_groups"][0]["paths"]) == 2
    assert report["near_duplicates"][0]["reason"] in ("phash", "phash_near", "near_crop")
    assert report["embedding"] == "unavailable"

    rows = [json.loads(line) for line in
            (cfg.output_dir / "metadata_staging.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    by_source = {Path(row["source_path"]).parent.name: row for row in rows}
    assert by_source["artist_a"]["artist"] == "Artist A"
    assert by_source["misc"]["artist"] == "unknown" and by_source["misc"]["artist_missing"]
    assert by_source["alice + bob"]["artists"] == ["alice", "bob"]
    for row in rows:
        assert row["staging_version"] == "v2"
        assert row["dedupe"]["status"] == "new"
        assert row["character"] == [] and row["character_missing"]

    images = list((cfg.output_dir / "images").iterdir())
    quarantine = list((cfg.output_dir / "quarantine").iterdir())
    assert len(images) == 3 and len(quarantine) == 1
    assert all(path.is_symlink() for path in images + quarantine)

    rejects = [json.loads(line) for line in
               (cfg.output_dir / "rejects.jsonl").read_text().splitlines()]
    reasons = {row["reason"] for row in rejects}
    assert {"gif", "video", "unsupported", "too_small", "blank", "bad_aspect",
            "unreadable", "exact_dup", "near_dup"} <= reasons
    assert not any(".hidden" in row["source_path"] for row in rejects)

    connection = sqlite3.connect(cfg.output_dir / "staging" / "index.sqlite")
    statuses = dict(connection.execute(
        "SELECT status, COUNT(*) FROM images GROUP BY status").fetchall())
    connection.close()
    assert statuses["accepted"] == 3 and statuses["near_dup"] == 1 and statuses["exact_dup"] == 1
    assert (cfg.output_dir / "dedupe_report.json").is_file()


def test_stage_pool_exact_dedupe_against_current_corpus(tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    current_image = save(current_dir / "current.png", noise_image(21))
    (current_dir / "metadata.jsonl").write_text(
        json.dumps({"file_name": "current.png"}) + "\n", encoding="utf-8")
    records = load_current_records(current_dir / "metadata.jsonl")
    assert len(records) == 1 and len(records[0].sha256) == 64

    pool = tmp_path / "pool"
    duplicate = pool / "artist_a/current_copy.png"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(current_image.read_bytes())
    save(pool / "artist_a/new.png", noise_image(22))
    cfg = config(tmp_path)
    report = stage_pool(cfg, current_records=records, progress=False)
    assert report["accepted"] == 1
    assert report["counts"]["exact_dup"] == 1
    assert report["exact_duplicate_groups"][0]["paths"][0].endswith("current/current.png")
    assert any(path.endswith("current_copy.png")
               for path in report["exact_duplicate_groups"][0]["paths"])


def test_staging_resume_skips_accepted(tmp_path):
    make_pool(tmp_path)
    cfg = config(tmp_path)
    first = stage_pool(cfg, artist_map={"artist a": "Artist A"},
                       known_artists={"alice", "bob"}, progress=False)
    assert first["accepted"] == 3
    cfg.resume = True
    second = stage_pool(cfg, artist_map={"artist a": "Artist A"},
                        known_artists={"alice", "bob"}, progress=False)
    assert second["counts"]["resumed"] == 4  # 3 accepted + 1 near-dup quarantine
    rows = (cfg.output_dir / "metadata_staging.jsonl").read_text().splitlines()
    assert len(rows) == 3


def test_hashrecord_dedupe_entry_compatibility():
    record = HashRecord("a" * 64, "0000000000000001", 1.0, "old.png")
    index = DedupeIndex([record])
    assert index.check(ImageInfo(64, 64, "a" * 64, "0" * 16), "new.png").status == "exact_dup"


def test_multi_artist_alias_is_mapped_before_known_check():
    resolution = resolve_artist("alias_a + bob/x.png",
                                artist_map={"alias a": "alice"},
                                known_artists={"alice", "bob"})
    assert resolution.artists == ("alice", "bob") and not resolution.ambiguous
