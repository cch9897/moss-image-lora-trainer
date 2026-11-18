"""VLM annotation driver: resume by image SHA, OCR context, atomic output."""

from __future__ import annotations

import json
from pathlib import Path

from moss_lora.vlm import VLMClient, VLMConfig
from scripts.vlm_annotate import ANNOTATIONS_NAME, annotate_metadata, read_jsonl, write_jsonl_atomic
from tests.helpers import gradient_image
from tests.test_vlm import combined_content, mock_server, response_body


def make_dataset(tmp_path: Path) -> tuple[Path, list[dict]]:
    images = tmp_path / "data"
    images.mkdir()
    gradient_image(256, 256, 1).save(images / "a.png")
    gradient_image(256, 256, 2).save(images / "b.png")
    rows = [
        {"file_name": "a.png", "width": 256, "height": 256, "bucket_w": 256, "bucket_h": 256,
         "artist": "alice", "tags": ["1girl", "city"],
         "layout_ocr": {"summary": {"caption": "ocr"}, "items": [
             {"label": "text", "caption": "visible text", "text": "EXIT",
              "bbox": [-100, -100, -50, 0]}],
             "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}},
        {"file_name": "b.png", "width": 256, "height": 256,
         "artist": "bob", "tags": ["city"]},
    ]
    metadata = images / "metadata_staging.jsonl"
    write_jsonl_atomic(metadata, rows)
    return metadata, rows


def test_annotate_and_resume_by_sha(tmp_path):
    metadata, rows = make_dataset(tmp_path)
    with mock_server([response_body(combined_content()), response_body(combined_content())]) as (url, server):
        client = VLMClient(VLMConfig(endpoint=url, model="gemma-test", model_family="gemma",
                                     backoff_base=0.0))
        report = annotate_metadata(rows, client, images_root=metadata.parent,
                                   output=tmp_path / "vlm", progress=False)
        assert report["records"] == 2 and report["statuses"]["ok"] == 2
        records = read_jsonl(tmp_path / "vlm" / ANNOTATIONS_NAME)
        assert len(records) == 2
        assert all(record["image_sha256"] and record["annotation_validated"] for record in records)
        first_prompt = server.requests[0]["messages"][1]["content"][1]["text"]
        assert "EXIT" in first_prompt and "bbox" in first_prompt

    with mock_server([]) as (url, server):
        client = VLMClient(VLMConfig(endpoint=url, model="gemma-test", model_family="gemma",
                                     backoff_base=0.0))
        report = annotate_metadata(rows, client, images_root=metadata.parent,
                                   output=tmp_path / "vlm", progress=False)
        assert report["skipped_resume"] == 2
        assert server.requests == []
        assert len(read_jsonl(tmp_path / "vlm" / ANNOTATIONS_NAME)) == 2


def test_force_reannotates_and_limit(tmp_path):
    metadata, rows = make_dataset(tmp_path)
    responses = [response_body(combined_content()), response_body(combined_content())]
    with mock_server(responses) as (url, _server):
        client = VLMClient(VLMConfig(endpoint=url, model="gemma-test", model_family="gemma",
                                     backoff_base=0.0))
        annotate_metadata(rows, client, images_root=metadata.parent,
                          output=tmp_path / "vlm", progress=False)
    with mock_server([response_body(combined_content())]) as (url, _server):
        client = VLMClient(VLMConfig(endpoint=url, model="gemma-test", model_family="gemma",
                                     backoff_base=0.0))
        report = annotate_metadata(rows, client, images_root=metadata.parent,
                                   output=tmp_path / "vlm", progress=False, force=True, limit=1)
        assert report["records"] == 2  # existing second row is preserved
        assert report["statuses"]["ok"] == 2
        assert len(read_jsonl(tmp_path / "vlm" / ANNOTATIONS_NAME)) == 2


def test_missing_image_is_recorded_not_fatal(tmp_path):
    metadata = tmp_path / "meta.jsonl"
    metadata.write_text(json.dumps({"file_name": "absent.png", "tags": []}) + "\n")
    with mock_server([]) as (url, _server):
        client = VLMClient(VLMConfig(endpoint=url, model="m", backoff_base=0.0))
        report = annotate_metadata([{"file_name": "absent.png"}], client,
                                   images_root=tmp_path, output=tmp_path / "vlm", progress=False)
    assert report["records"] == 1
    records = read_jsonl(tmp_path / "vlm" / ANNOTATIONS_NAME)
    assert records[0]["status"] == "invalid"
