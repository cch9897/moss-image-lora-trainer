"""Merge tests: strict SHA join, booru precedence, tag union/cap, reports, CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.merge_booru import (
    index_booru,
    main,
    merge_metadata,
    merge_row,
    render_markdown,
    row_sha256,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def hit_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sha256": SHA_A,
        "hit": True,
        "status": "hit",
        "characters": ["booru_hero"],
        "tags": ["booru_tag"],
        "rating": "s",
        "post_id": 4242,
        "source": "https://www.pixiv.net/artworks/4242",
        "source_host": "pixiv.net",
        "booru_artist": ["booru_alt"],
        "artist_match": False,
        "md5": "0" * 32,
        "api_md5": "1" * 32,
        "new_character": ["booru_hero"],
        "new_tags": ["booru_tag"],
    }
    row.update(overrides)
    return row


def base_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sha256": SHA_A,
        "source_path": "alice/001.png",
        "artist": "alice",
        "artists": ["alice"],
        "character": ["tagger_hero"],
        "tags": ["tagger_tag"],
        "rating": "general",
        "width": 832,
        "height": 1216,
        "caption_original": "a synthetic caption",
    }
    row.update(overrides)
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------- join key


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param({"sha256": SHA_A}, SHA_A, id="sha256"),
        pytest.param({"image_sha256": SHA_B}, SHA_B, id="image_sha256_fallback"),
        pytest.param({"sha256": "not-a-sha", "image_sha256": SHA_B}, SHA_B, id="invalid_then_image"),
        pytest.param({"sha256": SHA_A.upper()}, SHA_A, id="uppercase_normalized"),
        pytest.param({"sha256": f" {SHA_A} "}, SHA_A, id="whitespace_stripped"),
        pytest.param({"sha256": "a" * 63}, None, id="too_short"),
        pytest.param({"sha256": "g" * 64}, None, id="non_hex"),
        pytest.param({"sha256": "alice/001.png"}, None, id="path_not_a_key"),
        pytest.param({"sha256": None, "image_sha256": "x" * 64}, None, id="invalid_both"),
        pytest.param({"source_path": SHA_A, "file_name": "001.png"}, None, id="path_fields_ignored"),
    ],
)
def test_row_sha256_is_a_strict_digest_join_key(
    row: dict[str, Any], expected: str | None
) -> None:
    assert row_sha256(row) == expected


@pytest.mark.parametrize(
    "order",
    [("hit", "miss"), ("miss", "hit")],
    ids=["hit-before-miss", "miss-before-hit"],
)
def test_index_booru_hit_supersedes_miss_and_counts_duplicates(order: tuple[str, str]) -> None:
    by_kind = {"hit": hit_row(), "miss": hit_row(hit=False, status="miss")}
    index, duplicates = index_booru([by_kind[kind] for kind in order])
    assert duplicates == 1
    assert index[SHA_A]["hit"] is True
    assert index[SHA_A]["status"] == "hit"


def test_index_booru_ignores_records_without_sha() -> None:
    index, duplicates = index_booru(
        [hit_row(), {"source_path": SHA_B, "hit": True, "status": "hit"}]
    )
    assert set(index) == {SHA_A}
    assert duplicates == 0


# --------------------------------------------------------------------------- precedence


def test_booru_hit_labels_win_and_tags_are_a_booru_first_union() -> None:
    base = base_row(
        characters=["tagger_hero"],
        tags=["shared_tag", "tagger_tag", "tagger_hero", "worst_quality", "general"],
        tag_confidence={
            "shared_tag": 0.9,
            "tagger_tag": 0.4,
            "booru_hero": 0.8,
            "stale_tag": 0.1,
        },
    )
    booru = hit_row(
        characters=["booru_hero"],
        tags=[
            "shared_tag",
            "booru_tag",
            "booru_hero",
            "booru_alt",
            "sensitive",
            "bad_anatomy",
        ],
        booru_artist=["booru_alt"],
    )
    merged = merge_row(base, booru)

    assert merged["character"] == ["booru_hero"]
    assert merged["characters"] == ["booru_hero"]
    assert merged["character_source"] == "booru"
    assert merged["character_missing"] is False
    # booru order first, cross-source dedupe; character/artist/rating/negative tags dropped
    assert merged["tags"] == ["shared_tag", "booru_tag", "tagger_tag"]
    assert merged["tag_source"] == "booru+tagger"
    assert merged["artist"] == "alice"
    assert merged["artists"] == ["alice"]
    assert merged["booru_artist"] == ["booru_alt"]
    assert merged["rating"] == "sensitive"
    assert merged["rating_source"] == "booru"
    assert merged["tag_confidence"] == {"shared_tag": 0.9, "tagger_tag": 0.4}

    assert merged["booru_hit"] is True
    assert merged["booru_unresolved"] is False
    assert merged["booru_status"] == "hit"
    assert merged["booru_post_id"] == 4242
    assert merged["booru_source"] == "https://www.pixiv.net/artworks/4242"
    assert merged["booru_source_host"] == "pixiv.net"
    assert merged["booru_artist_match"] is False
    assert merged["booru_md5"] == "0" * 32
    assert merged["booru_api_md5"] == "1" * 32
    assert merged["booru_new_character"] == ["booru_hero"]
    assert merged["booru_new_tags"] == ["booru_tag"]


@pytest.mark.parametrize(
    ("record", "expected_status"),
    [
        pytest.param(
            hit_row(
                hit=False,
                status="miss",
                characters=["booru_hero"],
                tags=["booru_tag", "booru_hero"],
                rating="e",
            ),
            "miss",
            id="miss",
        ),
        pytest.param(
            hit_row(
                hit=False,
                status="error",
                characters=["booru_hero"],
                tags=["booru_tag", "booru_hero"],
                rating="e",
            ),
            "error",
            id="error",
        ),
        pytest.param({"sha256": SHA_A, "hit": False}, "unknown", id="status-unknown"),
        pytest.param(None, "no_record", id="no-record"),
    ],
)
def test_unresolved_booru_keeps_tagger_labels(
    record: dict[str, Any] | None, expected_status: str
) -> None:
    base = base_row(character=["tagger_hero"], tags=["tagger_tag"], rating="questionable")
    merged = merge_row(base, record)

    assert merged["character"] == ["tagger_hero"]
    assert merged["character_source"] == "tagger"
    assert merged["character_missing"] is False
    assert merged["tags"] == ["tagger_tag"]
    assert merged["tag_source"] == "tagger"
    assert merged["rating"] == "questionable"
    assert merged["rating_source"] == "tagger"
    assert merged["booru_hit"] is False
    assert merged["booru_unresolved"] is True
    assert merged["booru_status"] == expected_status
    assert merged["booru_post_id"] is None
    assert merged["booru_source"] is None
    assert merged["booru_source_host"] is None
    assert merged["booru_artist"] == []
    assert merged["booru_api_md5"] is None
    assert merged["booru_new_character"] == []
    assert merged["booru_new_tags"] == []


def test_booru_hit_without_characters_falls_back_to_tagger() -> None:
    merged = merge_row(
        base_row(character=["tagger_hero"], tags=["tagger_tag"]),
        hit_row(characters=[], tags=["booru_tag"]),
    )
    assert merged["character"] == ["tagger_hero"]
    assert merged["character_source"] == "tagger"
    assert merged["character_missing"] is False
    assert merged["tags"] == ["booru_tag", "tagger_tag"]
    assert merged["tag_source"] == "booru+tagger"
    assert merged.get("characters") is None


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(hit_row(characters=[], tags=["booru_tag"]), id="hit-empty"),
        pytest.param(hit_row(hit=False, status="miss"), id="miss"),
        pytest.param(None, id="no-record"),
    ],
)
def test_character_missing_only_when_both_sources_are_empty(
    record: dict[str, Any] | None,
) -> None:
    merged = merge_row(base_row(character=[], tags=[]), record)
    assert merged["character"] == []
    assert merged["character_source"] == "none"
    assert merged["character_missing"] is True


# --------------------------------------------------------------------------- tags


def test_tag_filtering_dedupes_and_reports_drop_counts() -> None:
    base = base_row(
        artist="alice-",
        artists=["alice-"],
        character=["tagger_hero"],
        tags=[
            "Shared_Tag",
            "tagger_tag",
            "tagger_hero",
            "alice",
            "worst_quality",
            "general",
        ],
        tag_confidence={
            "shared_tag": 0.9,
            "tagger_tag": 0.4,
            "blue_eyes": 0.05,
            "booru_hero": 0.8,
            "alice": 0.2,
            "stale_tag": 0.1,
        },
    )
    booru = hit_row(
        characters=["booru_hero"],
        tags=[
            "shared_tag",
            "blue_eyes",
            "Blue Eyes",
            "booru_hero",
            "booru_alt",
            "sensitive",
            "bad_anatomy",
        ],
        booru_artist=["booru_alt-"],
    )
    merged = merge_row(base, booru)

    assert merged["tags"] == ["shared_tag", "blue_eyes", "tagger_tag"]
    assert merged["tag_source"] == "booru+tagger"
    assert merged["tag_confidence"] == {
        "shared_tag": 0.9,
        "tagger_tag": 0.4,
        "blue_eyes": 0.05,
    }
    conflicts = merged["tag_conflicts"]
    assert conflicts["shared"] == 1
    assert conflicts["booru_only"] == 5
    assert conflicts["tagger_only"] == 5
    assert conflicts["selected_booru"] == 2
    assert conflicts["selected_tagger"] == 1
    assert conflicts["duplicate"] == 1
    assert conflicts["negative"] == 2
    assert conflicts["forbidden_identity_or_rating"] == 6
    assert conflicts.get("cap", 0) == 0


def test_tag_source_follows_surviving_origins_on_a_booru_hit() -> None:
    merged = merge_row(
        base_row(character=[], tags=["tagger_tag"]),
        hit_row(
            characters=[],
            tags=["sensitive", "general", "worst_quality"],
        ),
    )
    assert merged["booru_hit"] is True
    assert merged["tags"] == ["tagger_tag"]
    assert merged["tag_source"] == "tagger"


@pytest.mark.parametrize(
    ("booru_tags", "tagger_tags", "max_tags", "expected", "selected_booru", "selected_tagger", "capped"),
    [
        pytest.param(
            ["b1", "b2", "b3", "b4"],
            ["t1", "t2"],
            3,
            ["b1", "b2", "b3"],
            3,
            0,
            3,
            id="booru-fills-cap",
        ),
        pytest.param(
            ["b1", "b2"],
            ["t1", "t2"],
            3,
            ["b1", "b2", "t1"],
            2,
            1,
            1,
            id="tagger-fills-remainder",
        ),
    ],
)
def test_tag_cap_prefers_booru_order(
    booru_tags: list[str],
    tagger_tags: list[str],
    max_tags: int,
    expected: list[str],
    selected_booru: int,
    selected_tagger: int,
    capped: int,
) -> None:
    merged = merge_row(
        base_row(character=[], tags=tagger_tags),
        hit_row(characters=[], tags=booru_tags),
        max_tags=max_tags,
    )
    assert merged["tags"] == expected
    conflicts = merged["tag_conflicts"]
    assert conflicts["selected_booru"] == selected_booru
    assert conflicts["selected_tagger"] == selected_tagger
    assert conflicts.get("cap", 0) == capped
    assert merged["tag_source"] == ("booru+tagger" if selected_tagger else "booru")


# --------------------------------------------------------------------------- rating


@pytest.mark.parametrize(
    ("record", "base_rating", "expected_rating", "expected_source"),
    [
        pytest.param(hit_row(rating="e"), "general", "explicit", "booru", id="booru-short"),
        pytest.param(
            hit_row(rating="questionable"), "general", "questionable", "booru", id="booru-long"
        ),
        pytest.param(
            hit_row(rating=None, rating_label="sensitive"),
            "general",
            "sensitive",
            "booru",
            id="booru-rating-label",
        ),
        pytest.param(
            hit_row(rating=None, rating_label=None),
            "general",
            "general",
            "tagger",
            id="hit-without-rating-falls-back",
        ),
        pytest.param(
            hit_row(rating="not-a-rating"), "general", "general", "tagger", id="invalid-booru-rating"
        ),
        pytest.param(
            hit_row(hit=False, status="miss", rating="e"),
            "general",
            "general",
            "tagger",
            id="miss-rating-ignored",
        ),
        pytest.param(None, None, None, "none", id="neither"),
    ],
)
def test_rating_precedence(
    record: dict[str, Any] | None,
    base_rating: str | None,
    expected_rating: str | None,
    expected_source: str,
) -> None:
    merged = merge_row(base_row(rating=base_rating), record)
    assert merged["rating"] == expected_rating
    assert merged["rating_source"] == expected_source


# --------------------------------------------------------------------------- metadata join


def test_merge_metadata_emits_one_row_per_base_with_strict_sha_join() -> None:
    base_rows = [
        base_row(sha256=SHA_A),
        base_row(sha256=None, source_path="alice/no-sha.png"),
        base_row(sha256="f" * 63, source_path="alice/short-sha.png"),
        base_row(sha256=None, image_sha256=SHA_B, source_path="alice/image-sha.png"),
    ]
    booru_rows = [
        hit_row(sha256=SHA_A, characters=["hero_a"], tags=["tag_a"]),
        hit_row(sha256=SHA_B, characters=["hero_b"], tags=["tag_b"]),
        hit_row(sha256=SHA_C, tags=["booru_only_tag"]),
        # Same digest as SHA_A but a miss; the earlier hit must survive.
        hit_row(sha256=SHA_A, hit=False, status="miss"),
        # Same path as a base row, but no digest: must not join.
        {"source_path": "alice/no-sha.png", "hit": True, "status": "hit", "tags": ["path_only"]},
    ]
    rows, report = merge_metadata(base_rows, booru_rows)

    assert [row["source_path"] for row in rows] == [
        "alice/001.png",
        "alice/no-sha.png",
        "alice/short-sha.png",
        "alice/image-sha.png",
    ]
    assert [row["booru_status"] for row in rows] == ["hit", "no_record", "no_record", "hit"]
    assert [row["booru_hit"] for row in rows] == [True, False, False, True]
    assert report["rows"] == 4
    assert report["booru_records"] == 3
    assert report["booru_duplicate_records"] == 1
    assert report["booru_only_records"] == 1
    assert report["booru_hit"] == 2
    assert report["booru_unresolved"] == 2


def test_limit_truncates_base_rows_but_reports_input_size() -> None:
    base_rows = [
        base_row(sha256=digest, source_path=f"alice/{index}.png")
        for index, digest in enumerate((SHA_A, SHA_B, SHA_C))
    ]
    booru_rows = [hit_row(sha256=digest) for digest in (SHA_A, SHA_B, SHA_C)]
    rows, report = merge_metadata(base_rows, booru_rows, limit=2)

    assert [row["source_path"] for row in rows] == ["alice/0.png", "alice/1.png"]
    assert report["rows"] == 2
    assert report["base_rows"] == 3
    assert report["booru_hit"] == 2


# --------------------------------------------------------------------------- preservation


def test_base_fields_and_layout_ocr_survive(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_annotations"
    raw_dir.mkdir()
    (raw_dir / f"{SHA_B}.json").write_text(
        json.dumps({"image_sha256": SHA_B, "ocr": {"items": [{"text": "raw_b"}]}}),
        encoding="utf-8",
    )
    (raw_dir / f"{SHA_C}.json").write_text(
        json.dumps({"image_sha256": SHA_C, "ocr": {"items": [{"text": "raw_c"}]}}),
        encoding="utf-8",
    )
    base_rows = [
        base_row(sha256=SHA_A),
        base_row(sha256=SHA_B, layout_ocr={"items": [{"text": "base_b"}]}),
        base_row(sha256=SHA_C),
        base_row(sha256=None, source_path="alice/ocr-by-path.png"),
    ]
    ocr_rows = [
        {"sha256": SHA_A, "layout_ocr": {"items": [{"text": "ocr_a"}]}},
        {"sha256": SHA_B, "layout_ocr": {"items": [{"text": "ocr_b"}]}},
        {"source_path": "alice/ocr-by-path.png", "layout": {"items": [{"text": "ocr_path"}]}},
    ]
    rows, report = merge_metadata(base_rows, [], ocr_rows=ocr_rows, raw_annotations=raw_dir)

    assert rows[0]["layout_ocr"] == {"items": [{"text": "ocr_a"}]}
    assert rows[1]["layout_ocr"] == {"items": [{"text": "base_b"}]}
    assert rows[2]["layout_ocr"] == {"items": [{"text": "raw_c"}]}
    assert rows[3]["layout_ocr"] == {"items": [{"text": "ocr_path"}]}
    assert report["layout_ocr_rows"] == 4
    for row, base in zip(rows, base_rows):
        for field in ("source_path", "artist", "artists", "width", "height", "caption_original"):
            assert row[field] == base[field]


# --------------------------------------------------------------------------- report


def test_report_aggregates_and_markdown_excludes_tag_values() -> None:
    base_rows = [
        base_row(
            sha256=SHA_A,
            artist="alice",
            artists=["alice"],
            character=["tagger_char"],
            tags=["tagger_x"],
            rating="general",
        ),
        base_row(
            sha256=SHA_B,
            artist="alice",
            artists=["alice"],
            character=["tagger_char2"],
            tags=["tagger_y"],
            rating=None,
        ),
        base_row(
            sha256=None,
            artist="bob",
            artists=["bob"],
            character=[],
            tags=[],
            rating=None,
        ),
    ]
    booru_rows = [
        hit_row(sha256=SHA_A, characters=["zzchar_alpha"], tags=["zztag_alpha"], rating="s"),
        hit_row(sha256=SHA_B, hit=False, status="miss"),
    ]
    _, report = merge_metadata(base_rows, booru_rows)

    assert report["rows"] == 3
    assert report["base_rows"] == 3
    assert report["booru_records"] == 2
    assert report["booru_duplicate_records"] == 0
    assert report["booru_only_records"] == 0
    assert report["booru_hit"] == 1
    assert report["booru_unresolved"] == 2
    assert report["booru_statuses"] == {"hit": 1, "miss": 1}
    assert report["character_from_booru"] == 1
    assert report["character_from_tagger"] == 1
    assert report["character_missing"] == 1
    assert report["tagger_fallback"] == 1
    assert report["tags_from_booru_only"] == 0
    assert report["tags_from_tagger_only"] == 1
    assert report["tags_from_both"] == 1
    assert report["tags_from_none"] == 1
    assert report["rating_from_booru"] == 1
    assert report["rating_from_tagger"] == 0
    assert report["rating_from_none"] == 2
    assert report["layout_ocr_rows"] == 0
    assert report["tag_conflicts"]["shared"] == 0
    assert report["tag_conflicts"]["booru_only"] == 1
    assert report["tag_conflicts"]["tagger_only"] == 2
    assert report["tag_conflicts"]["selected_booru"] == 1
    assert report["tag_conflicts"]["selected_tagger"] == 2

    assert report["by_artist"]["alice"]["rows"] == 2
    assert report["by_artist"]["alice"]["booru_hit"] == 1
    assert report["by_artist"]["alice"]["character_from_booru"] == 1
    assert report["by_artist"]["alice"]["character_from_tagger"] == 1
    assert report["by_artist"]["alice"]["tags_from_both"] == 1
    assert report["by_artist"]["alice"]["tags_from_tagger_only"] == 1
    assert report["by_artist"]["bob"]["rows"] == 1
    assert report["by_artist"]["bob"]["character_missing"] == 1
    assert report["by_artist"]["bob"]["tags_from_none"] == 1
    assert report["hits_by_artist"] == {"alice": 1, "bob": 0}

    markdown = render_markdown(report)
    for value in (
        "zztag_alpha",
        "tagger_x",
        "tagger_y",
        "zzchar_alpha",
        "tagger_char",
        "tagger_char2",
    ):
        assert value not in markdown
    assert "| Rows | 3 |" in markdown
    assert "| Booru hit | 1 |" in markdown


# --------------------------------------------------------------------------- CLI


def test_main_writes_jsonl_report_and_markdown_with_counts_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = write_jsonl(
        tmp_path / "metadata_tagged.jsonl",
        [
            base_row(sha256=SHA_A, tags=["zztag_base_a"]),
            base_row(sha256=SHA_B, tags=["zztag_base_b"]),
            base_row(
                sha256=None,
                source_path="alice/no-sha.png",
                character=[],
                tags=["zztag_base_c"],
                rating=None,
            ),
        ],
    )
    booru_path = write_jsonl(
        tmp_path / "booru_recovered.jsonl",
        [
            hit_row(
                sha256=SHA_A,
                characters=["zzchar_alpha"],
                tags=["zztag_booru_a"],
                rating="e",
            ),
            hit_row(sha256=SHA_B, hit=False, status="miss"),
        ],
    )
    ocr_path = write_jsonl(
        tmp_path / "ocr.jsonl",
        [{"sha256": SHA_A, "layout_ocr": {"items": [{"text": "zz_ocr_a"}]}}],
    )
    raw_dir = tmp_path / "raw_annotations"
    raw_dir.mkdir()
    (raw_dir / f"{SHA_B}.json").write_text(
        json.dumps({"image_sha256": SHA_B, "ocr": {"items": [{"text": "zz_ocr_b"}]}}),
        encoding="utf-8",
    )
    out = tmp_path / "merged.jsonl"
    report_path = tmp_path / "merged_report.json"
    md_path = tmp_path / "merged_report.md"

    code = main(
        [
            "--input",
            str(input_path),
            "--booru",
            str(booru_path),
            "--out",
            str(out),
            "--report",
            str(report_path),
            "--md-report",
            str(md_path),
            "--ocr",
            str(ocr_path),
            "--raw-annotations",
            str(raw_dir),
        ]
    )

    assert code == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["booru_status"] == "hit"
    assert rows[0]["tags"] == ["zztag_booru_a", "zztag_base_a"]
    assert rows[0]["rating"] == "explicit"
    assert rows[0]["rating_source"] == "booru"
    assert rows[0]["layout_ocr"] == {"items": [{"text": "zz_ocr_a"}]}
    assert rows[1]["booru_status"] == "miss"
    assert rows[1]["tags"] == ["zztag_base_b"]
    assert rows[1]["layout_ocr"] == {"items": [{"text": "zz_ocr_b"}]}
    assert rows[2]["booru_status"] == "no_record"
    assert rows[2]["character_missing"] is True

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["rows"] == 3
    assert report["booru_hit"] == 1
    markdown = md_path.read_text(encoding="utf-8")
    for value in ("zztag_booru_a", "zztag_base_a", "zztag_base_b", "zzchar_alpha"):
        assert value not in markdown

    captured = capsys.readouterr()
    summary = json.loads(captured.err.strip().splitlines()[-1])
    assert summary["rows"] == 3
    assert summary["booru_hit"] == 1
    assert summary["character_from_booru"] == 1
    assert summary["character_from_tagger"] == 1
    assert summary["character_missing"] == 1
    assert summary["report"] == str(report_path)
    assert summary["out"] == str(out)
    assert "zztag" not in captured.err
    assert "zzchar" not in captured.err


def test_main_returns_nonzero_for_unreadable_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    booru_path = write_jsonl(tmp_path / "booru_recovered.jsonl", [])
    out = tmp_path / "merged.jsonl"
    missing = tmp_path / "missing.jsonl"

    code = main(["--input", str(missing), "--booru", str(booru_path), "--out", str(out)])

    assert code == 1
    assert not out.exists()
    assert not (tmp_path / "merged_report.json").exists()
    error = capsys.readouterr().err
    assert error.startswith("error:")
    assert str(missing) in error
