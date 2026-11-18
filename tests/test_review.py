"""Layout review tests: geometry, path safety, queue, save/export, and the HTTP API."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from moss_lora.cache import assign_bucket, build_buckets
from moss_lora.review import (
    ReviewError,
    ReviewStore,
    bbox_to_bucket_rect,
    bucket_rect_to_bbox,
    coerce_bbox,
    create_app,
    layout_items,
    layout_of,
    prose_long,
    resolve_image_path,
    text_hint,
)

TARGET_AREA = 262_144
POSTER_PROSE = "A poster on a wall, with the word EXIT in white letters."


@dataclass(frozen=True)
class Dataset:
    root: Path
    images: Path
    metadata: Path
    output: Path
    rows: list[dict]


def _write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


@pytest.fixture()
def dataset(tmp_path: Path) -> Dataset:
    root = tmp_path / "dataset"
    images = root / "images"
    _write_image(images / "a.png", (900, 1200), (200, 180, 150))
    _write_image(images / "b.png", (1200, 700), (120, 160, 190))
    _write_image(images / "c.png", (700, 900), (150, 190, 120))
    _write_image(images / "d.png", (900, 1200), (180, 150, 200))
    _write_image(images / "sub" / "a.png", (900, 1200), (200, 180, 150))

    bucket = assign_bucket(900, 1200, build_buckets(TARGET_AREA))
    layout_bbox = bucket_rect_to_bbox(20, 30, 120, 60, bucket)

    rows = [
        {
            "file_name": "a.png",
            "artist": "alice",
            "character": "hero",
            "caption": {
                "long": POSTER_PROSE,
                "medium": "poster, exit sign",
                "short": "poster",
            },
            "caption_no_artist": "A poster without the artist tag.",
            "tags": ["poster", "sign"],
            "layout": {
                "summary": {"meta": ["anime"], "canvas": "single",
                            "caption": "A poster on a wall."},
                "items": [
                    {"label": "text", "caption": "visible text", "text": "EXIT",
                     "bbox": layout_bbox},
                ],
                "bbox_coord_space": "abs_centralized",
                "bbox_format": "yxyx",
            },
        },
        {
            "file_name": "b.png",
            "artist": "bob",
            "character": ["hero", "mage"],
            "caption": {
                "long": 'A quiet room with a sign reading "EXIT".',
                "medium": "quiet room",
                "short": "room",
            },
            "caption_no_artist": "A quiet room.",
            "tags": ["room", "sign"],
        },
        {
            "file_name": "c.png",
            "artist": "carol",
            "character": "cat",
            "caption": {
                "long": "a plain field under a clear sky",
                "medium": "a plain field",
                "short": "field",
            },
            "caption_no_artist": "a plain field",
            "tags": ["landscape"],
        },
        {
            "file_name": "d.png",
            "artist": "dave",
            "character": "bird",
            "caption": {
                "long": "a calm scene",
                "medium": "calm scene",
                "short": "scene",
            },
            "caption_no_artist": "a calm scene",
            "tags": ["writing"],
        },
    ]

    metadata = root / "metadata.jsonl"
    metadata.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return Dataset(root=root, images=images, metadata=metadata,
                   output=tmp_path / "review", rows=rows)


@pytest.fixture()
def store(dataset: Dataset) -> ReviewStore:
    return ReviewStore(dataset.metadata, dataset.images, dataset.output,
                       target_area=TARGET_AREA)


@pytest.fixture()
def client(store: ReviewStore):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    return TestClient(create_app(store, initial_filter="text"))


def reviewed_rows(dataset: Dataset) -> list[dict]:
    text = (dataset.output / "metadata_reviewed.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------- geometry


@pytest.mark.parametrize("bucket", [(448, 576), (704, 384), (512, 512), (1024, 1024)])
@pytest.mark.parametrize("rect", [(0, 0, 64, 64), (17, 31, 96, 48), (1, 2, 3, 4)])
def test_bucket_rect_bbox_roundtrip_stays_integer(
    bucket: tuple[int, int], rect: tuple[int, int, int, int],
) -> None:
    x, y, w, h = rect
    bbox = bucket_rect_to_bbox(x, y, w, h, bucket)

    half_w, half_h = bucket[0] // 2, bucket[1] // 2
    assert bbox == [y - half_h, x - half_w, y + h - half_h, x + w - half_w]
    assert all(type(value) is int for value in bbox)
    assert bbox_to_bucket_rect(bbox, bucket) == (x, y, w, h)


@pytest.mark.parametrize("bbox", [
    [-288, -224, 288, 224],  # exact bucket boundary for a 448x576 bucket
    [-288, -224, -1, -1],
    [0, 0, 1, 1],
])
def test_coerce_bbox_accepts_in_bounds_boxes(bbox: list[int]) -> None:
    assert coerce_bbox(bbox, (448, 576)) == bbox


def test_coerce_bbox_normalizes_integral_floats() -> None:
    normalized = coerce_bbox([0.0, 1.0, 2.0, 3.0], (448, 576))
    assert normalized == [0, 1, 2, 3]
    assert all(type(value) is int for value in normalized)


@pytest.mark.parametrize(("bbox", "message"), [
    ([-289, 0, 10, 10], "outside the bucket"),
    ([0, -225, 10, 10], "outside the bucket"),
    ([0, 0, 289, 10], "outside the bucket"),
    ([0, 0, 10, 225], "outside the bucket"),
    ([10, 10, 10, 20], "positive width"),
    ([10, 10, 20, 10], "positive width"),
    ([0.5, 0, 10, 10], "integers"),
    ([True, 0, 10, 10], "integers"),
    (["0", 0, 10, 10], "integers"),
    ([0, 0, 10], "four integers"),
])
def test_coerce_bbox_rejects_invalid_boxes(bbox, message: str) -> None:
    with pytest.raises(ReviewError, match=message):
        coerce_bbox(bbox, (448, 576))


# --------------------------------------------------------------------------- paths


def test_resolve_image_path_supports_metadata_and_image_roots(dataset: Dataset) -> None:
    assert resolve_image_path(dataset.images, dataset.root, "a.png") == (
        dataset.images.resolve() / "a.png"
    )
    assert resolve_image_path(dataset.root, dataset.root, "images/a.png") == (
        dataset.images.resolve() / "a.png"
    )
    assert resolve_image_path(dataset.images, dataset.root, "sub/a.png") == (
        dataset.images.resolve() / "sub" / "a.png"
    )


def test_resolve_image_path_follows_symlinks(dataset: Dataset) -> None:
    link = dataset.images / "link.png"
    link.symlink_to(dataset.images / "a.png")

    resolved = resolve_image_path(dataset.images, dataset.root, "link.png")

    assert resolved.name == "link.png"
    assert resolved.is_symlink()
    assert resolved.is_file()


@pytest.mark.parametrize("name", [
    "../secret.png",
    "sub/../../secret.png",
    "/tmp/secret.png",
    "sub\\a.png",
    "notes.txt",
    "",
    "a\x00.png",
])
def test_resolve_image_path_rejects_unsafe_names(dataset: Dataset, name: str) -> None:
    with pytest.raises(ReviewError):
        resolve_image_path(dataset.images, dataset.root, name)


def test_resolve_image_path_cannot_escape_through_metadata_dir(
    dataset: Dataset, tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    _write_image(outside / "evil.png", (64, 64), (255, 0, 0))

    with pytest.raises(ReviewError, match="not found"):
        resolve_image_path(dataset.images, outside, "evil.png")


# --------------------------------------------------------------------------- queue


@pytest.mark.parametrize(("row", "expected"), [
    ({"caption": {"long": "a sign above the door"}}, True),
    ({"caption": {"long": "a speech bubble near her face"}}, True),
    ({"caption": {"long": "some writing on the wall"}}, True),
    ({"caption": {"long": "a page with text"}}, True),
    ({"caption": {"long": 'a board that says "EXIT"'}}, True),
    ({"tags": ["lettering"]}, True),
    ({"caption": {"long": "a quiet room"}}, False),
    ({"caption": {"long": "a plain field under a clear sky"}}, False),
])
def test_text_hint_signals(row: dict, expected: bool) -> None:
    assert text_hint(row) is expected


def test_queue_filter_and_counts(store: ReviewStore) -> None:
    listing = store.list_items("text")
    by_index = {item["index"]: item for item in listing["items"]}

    assert sorted(by_index) == [0, 1, 3]
    assert by_index[0]["reason"] == "layout"
    assert by_index[0]["has_layout"] is True
    assert by_index[0]["layout_items"] == 1
    assert by_index[1]["reason"] == "text_no_layout"
    assert by_index[3]["reason"] == "text_no_layout"

    everything = store.list_items("all")
    all_by_index = {item["index"]: item for item in everything["items"]}
    assert sorted(all_by_index) == [0, 1, 2, 3]
    assert all_by_index[2]["in_queue"] is False
    assert all_by_index[2]["reason"] is None

    counts = store.counts()
    assert counts["total"] == 4
    assert counts["queue"] == 3
    assert counts["text_no_layout"] == 2
    assert counts["layouts"] == 1
    assert counts["remaining_queue"] == 3
    assert counts["boxes"] == 1
    assert counts["completed"] == 0
    assert listing["counts"]["filtered"] == 3
    assert everything["counts"]["filtered"] == 4

    original = store.records[0]
    assert layout_items(layout_of(original.row))[0]["text"] == "EXIT"
    assert prose_long(original.row) == POSTER_PROSE


# --------------------------------------------------------------------------- save


def test_save_add_edit_delete_and_resume(dataset: Dataset, store: ReviewStore) -> None:
    detail = store.detail(0)
    bucket = (detail["bucket"]["width"], detail["bucket"]["height"])
    first_bbox = detail["boxes"][0]["bbox"]
    second_bbox = bucket_rect_to_bbox(20, 40, 100, 60, bucket)
    assert coerce_bbox(second_bbox, bucket) == second_bbox

    saved = store.save(0, [
        {"bbox": first_bbox, "text": "EXIT"},
        {"bbox": second_bbox, "text": "OPEN"},
    ])

    assert saved["status"] == "edited"
    assert [box["text"] for box in saved["boxes"]] == ["EXIT", "OPEN"]
    row = reviewed_rows(dataset)[0]
    assert row["caption"]["medium"] == dataset.rows[0]["caption"]["medium"]
    assert row["caption"]["short"] == dataset.rows[0]["caption"]["short"]
    layout = json.loads(row["caption"]["long"])
    assert layout == row["layout"]
    assert layout["bbox_format"] == "yxyx"
    assert layout["bbox_coord_space"] == "abs_centralized"
    assert [item["text"] for item in layout_items(layout)] == ["EXIT", "OPEN"]
    for item in layout_items(layout):
        assert coerce_bbox(item["bbox"], bucket) == item["bbox"]
    summary = layout["summary"]["caption"]
    assert "artist:alice" in summary
    assert "character:hero" in summary
    assert store.reviewed_path.is_file()
    assert store.state_path.is_file()
    assert store.report_path.is_file()

    source = [json.loads(line) for line in
              dataset.metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert source[0]["caption"]["long"] == POSTER_PROSE
    assert [item["text"] for item in source[0]["layout"]["items"]] == ["EXIT"]

    remaining = [{"bbox": second_bbox, "text": "OPEN"}]
    saved = store.save(0, remaining)
    assert saved["status"] == "edited"
    assert saved["boxes"] == remaining
    row = reviewed_rows(dataset)[0]
    layout = layout_of(row)
    assert layout is not None
    assert [item["bbox"] for item in layout_items(layout)] == [second_bbox]
    assert json.loads(row["caption"]["long"])["items"] == row["layout"]["items"]

    resumed = ReviewStore(dataset.metadata, dataset.images, dataset.output,
                          target_area=TARGET_AREA)
    assert resumed.loaded_from_reviewed is True
    resumed_detail = resumed.detail(0)
    assert resumed_detail["status"] == "edited"
    assert resumed_detail["boxes"] == remaining
    assert [item["text"] for item in resumed_detail["layout"]["items"]] == ["OPEN"]
    assert resumed.counts()["boxes"] == 1


def test_no_text_clears_layout_and_restores_prose(dataset: Dataset, store: ReviewStore) -> None:
    detail = store.detail(0)
    bucket = (detail["bucket"]["width"], detail["bucket"]["height"])
    bbox = bucket_rect_to_bbox(10, 10, 50, 30, bucket)
    store.save(0, [{"bbox": bbox, "text": "EXIT"}])
    assert "layout" in reviewed_rows(dataset)[0]
    assert reviewed_rows(dataset)[0]["caption"]["long"].startswith("{")

    saved = store.save(0, [], no_text=True)

    assert saved["status"] == "no_text"
    assert saved["no_text"] is True
    assert saved["layout"] is None
    assert saved["boxes"] == []
    row = reviewed_rows(dataset)[0]
    assert "layout" not in row
    assert row["caption"]["long"] == dataset.rows[0]["caption"]["long"] == POSTER_PROSE
    assert row["caption"]["medium"] == dataset.rows[0]["caption"]["medium"]
    assert row["caption"]["short"] == dataset.rows[0]["caption"]["short"]
    assert row["caption_no_artist"] == dataset.rows[0]["caption_no_artist"]
    assert reviewed_rows(dataset)[1] == dataset.rows[1]

    state = json.loads(store.state_path.read_text(encoding="utf-8"))
    entry = state["items"]["0"]
    assert entry["file_name"] == "a.png"
    assert entry["status"] == "no_text"
    assert entry["boxes"] == 0
    assert entry["no_text"] is True
    assert entry["updated_at"]

    report = store.report_path.read_text(encoding="utf-8")
    assert "| marked no text | 1 |" in report


# --------------------------------------------------------------------------- API


def test_api_read_surface_and_bucket_image(client) -> None:
    index_page = client.get("/")
    assert index_page.status_code == 200
    assert index_page.headers["content-type"].startswith("text/html")
    assert index_page.headers["cache-control"] == "no-store, max-age=0"

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    listing = client.get("/api/items", params={"filter": "text"})
    assert listing.status_code == 200
    assert [item["index"] for item in listing.json()["items"]] == [0, 1, 3]
    assert listing.json()["counts"]["queue"] == 3
    assert listing.headers["cache-control"] == "no-store, max-age=0"

    detail_response = client.get("/api/items/0")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["file_name"] == "a.png"
    assert detail["boxes"][0]["text"] == "EXIT"

    second = client.get("/api/items/1").json()
    bucket = (second["bucket"]["width"], second["bucket"]["height"])
    assert bucket[0] != bucket[1]  # the 1200x700 source must stay non-square
    image_response = client.get("/api/image/1")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/jpeg"
    assert image_response.headers["cache-control"] == "no-store, max-age=0"
    with Image.open(io.BytesIO(image_response.content)) as image:
        assert image.format == "JPEG"
        assert image.size == bucket


def test_api_post_save_persists_edit(client) -> None:
    detail = client.get("/api/items/0").json()
    bucket = (detail["bucket"]["width"], detail["bucket"]["height"])
    bbox = bucket_rect_to_bbox(30, 50, 120, 80, bucket)

    response = client.post(
        "/api/items/0/save",
        json={"boxes": [{"bbox": bbox, "text": "OPEN"}]},
    )

    assert response.status_code == 200
    saved = response.json()
    assert saved["status"] == "edited"
    assert saved["boxes"] == [{"bbox": bbox, "text": "OPEN"}]
    assert saved["layout"]["items"][0]["text"] == "OPEN"
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert client.get("/api/items/0").json()["status"] == "edited"


def test_api_export_and_error_contracts(client) -> None:
    exported = client.post("/api/export")
    assert exported.status_code == 200
    paths = exported.json()
    for key in ("metadata", "state", "report"):
        assert Path(paths[key]).is_file()

    detail = client.get("/api/items/0").json()
    bucket = (detail["bucket"]["width"], detail["bucket"]["height"])
    valid_bbox = bucket_rect_to_bbox(30, 50, 120, 80, bucket)
    outside = client.post(
        "/api/items/0/save",
        json={"boxes": [{"bbox": [0, 0, 10**6, 10**6], "text": "X"}]},
    )
    assert outside.status_code == 400
    assert outside.headers["content-type"].startswith("application/json")
    assert "error" in outside.json()

    empty_text = client.post(
        "/api/items/0/save",
        json={"boxes": [{"bbox": valid_bbox, "text": "   "}]},
    )
    assert empty_text.status_code == 400
    assert "error" in empty_text.json()

    for path in ("/api/items/999", "/api/image/999", "/metadata.jsonl"):
        missing = client.get(path)
        assert missing.status_code == 404
        assert missing.headers["content-type"].startswith("application/json")
        assert missing.headers["cache-control"] == "no-store, max-age=0"
        assert "error" in missing.json()


def test_review_preserves_object_and_character_labels(dataset: Dataset, store: ReviewStore) -> None:
    detail = store.detail(0)
    bucket = (detail["bucket"]["width"], detail["bucket"]["height"])
    text_bbox = bucket_rect_to_bbox(20, 30, 120, 60, bucket)
    object_bbox = bucket_rect_to_bbox(180, 120, 110, 90, bucket)
    saved = store.save(0, [
        {"bbox": text_bbox, "label": "text", "text": "EXIT"},
        {"bbox": object_bbox, "label": "character", "caption": "the hero standing"},
    ])
    row = reviewed_rows(dataset)[0]
    items = row["layout"]["items"]
    assert [item["label"] for item in items] == ["character", "text"] or [
        item["label"] for item in items] == ["text", "character"]
    character = next(item for item in items if item["label"] == "character")
    assert character["caption"] == "the hero standing" and "text" not in character
    assert any(box.get("label") == "character" and box.get("caption") == "the hero standing"
               for box in saved["boxes"])
    assert saved["layout_policy"] == "legacy"


def test_review_separate_layout_policy_keeps_prose_long(dataset: Dataset) -> None:
    separated = ReviewStore(dataset.metadata, dataset.images, dataset.output / "separate",
                            target_area=TARGET_AREA, layout_policy="separate")
    detail = separated.detail(0)
    bucket = (detail["bucket"]["width"], detail["bucket"]["height"])
    bbox = bucket_rect_to_bbox(20, 30, 120, 60, bucket)
    separated.save(0, [{"bbox": bbox, "text": "EXIT"}])
    row = json.loads(separated.reviewed_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["caption"]["long"] == POSTER_PROSE
    assert row["layout"]["items"][0]["text"] == "EXIT"
    assert separated.detail(0)["layout_policy"] == "separate"
    assert separated.detail(0)["artists"] == ["alice"]


def test_review_rejects_unknown_label(dataset: Dataset, store: ReviewStore) -> None:
    detail = store.detail(0)
    bucket = (detail["bucket"]["width"], detail["bucket"]["height"])
    bbox = bucket_rect_to_bbox(20, 30, 120, 60, bucket)
    with pytest.raises(ReviewError):
        store.save(0, [{"bbox": bbox, "label": "bogus", "caption": "x"}])
