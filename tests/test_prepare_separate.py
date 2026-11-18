"""prepare_captions separate layout mode and v2 metadata fields."""

from __future__ import annotations

import json
from pathlib import Path

from moss_lora.cache import caption_variant_records, collect_items
from scripts import prepare_captions
from tests.test_ocr_layout import config, fake_ocr, load_row, write_case


def prepare(tmp_path: Path, monkeypatch, **overrides) -> dict:
    root = tmp_path / "data"
    write_case(root)
    monkeypatch.setattr(prepare_captions, "backend_available", lambda backend: True)
    monkeypatch.setattr(prepare_captions, "run_ocr", fake_ocr())
    output = tmp_path / "out"
    prepare_captions.prepare_dataset(root, output, config(**overrides))
    return load_row(output)


def test_separate_keeps_long_prose_and_emits_layout_fields(tmp_path, monkeypatch):
    row = prepare(tmp_path, monkeypatch, layout_mode="separate")
    original = row["caption_original"]
    assert original.startswith("mystyle, artist:alice, character:hero")
    assert row["caption"]["long"] == original
    assert not row["caption"]["long"].lstrip().startswith("{")
    assert row["layout"]["items"][0]["text"] == "EXIT"
    assert row["layout_ocr"] == row["layout"]
    assert row["artists"] == ["alice"]

    raw = collect_items(tmp_path / "out")[0][1]
    records = caption_variant_records(raw)
    assert records[0].kind == "original" and records[-1].kind == "layout"
    assert "artist:alice" in records[0].text
    layout_variant = json.loads(records[-1].text)
    assert layout_variant["items"][0]["text"] == "EXIT"


def test_legacy_both_mode_still_puts_layout_in_long(tmp_path, monkeypatch):
    row = prepare(tmp_path, monkeypatch, layout_mode="both")
    assert row["caption"]["long"].lstrip().startswith('{"summary"')
    assert row["caption_original"].startswith("mystyle, artist:alice")
    assert row["layout_ocr"] == row["layout"]


def test_field_mode_keeps_prose_and_layout_field(tmp_path, monkeypatch):
    row = prepare(tmp_path, monkeypatch, layout_mode="field")
    assert not row["caption"]["long"].lstrip().startswith("{")
    assert row["layout"]["items"][0]["text"] == "EXIT"
    assert row["layout_ocr"] == row["layout"]


def test_separate_alias_artist_metadata(tmp_path, monkeypatch):
    row = prepare(tmp_path, monkeypatch, layout_mode="separate",
                  artist_map={"alice": "Alice Canonical"})
    assert row["artists"] == ["Alice Canonical"]
    assert row["artist"] == "Alice Canonical"
    assert "artist:Alice Canonical" in row["caption_original"]
