"""Tests for the read-only dashboard: discovery, parsing, status, security and HTTP API."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from moss_lora.dashboard import RunStore

METRICS_HEADER = "step,loss,lr,sigma,grad_norm,sec_per_step,peak_vram_gb"


# --------------------------------------------------------------------------- fixtures


def write_metrics(run: Path, rows: list[str], header: str = METRICS_HEADER) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "metrics.csv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def write_png(path: Path, color=(197, 59, 44)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), color).save(path)


def write_state(run: Path, *, step: int = 50, max_steps: int = 100, config: dict | None = None) -> None:
    payload = {
        "step": step,
        "epoch": 2,
        "batch_index": 10,
        "moss_lora_version": "1.0.0",
        "config": config
        if config is not None
        else {
            "model": {"path": "models/MOSS-Image-Base", "dtype": "bf16", "revision": "abc123"},
            "data": {"cache_dir": "cache/1m", "batch_size": 1, "num_workers": 4},
            "train": {
                "max_steps": max_steps,
                "learning_rate": 1e-4,
                "warmup_steps": 50,
                "lr_scheduler": "cosine",
                "seed": 42,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": True,
                "compile": False,
                "ema_decay": 0.0,
            },
            "lora": {"rank": 32, "alpha": None, "targets": ["q", "k"]},
            "logging": {"save_every": 200, "log_every": 10, "sample_every": 250, "sample_steps": 30},
        },
    }
    (run / "trainer_state.json").write_text(json.dumps(payload), encoding="utf-8")


def write_checkpoint(run: Path, step: int, *, adapter_size: int = 100, optimizer_size: int = 200) -> Path:
    directory = run / "checkpoints" / f"step_{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_model.safetensors").write_bytes(b"a" * adapter_size)
    (directory / "adapter_config.json").write_text(
        json.dumps({"r": 16, "alpha": 32, "targets": ["q", "v"], "base_model": "models/MOSS-Image-Base"}),
        encoding="utf-8",
    )
    (directory / "optimizer.pt").write_bytes(b"o" * optimizer_size)
    (directory / "lr_scheduler.pt").write_bytes(b"s")
    (directory / "rng.pt").write_bytes(b"r")
    (directory / "trainer_state.json").write_text(
        json.dumps({"step": step, "epoch": 2, "batch_index": 10,
                    "config": {"train": {"max_steps": 100}}}),
        encoding="utf-8",
    )
    return directory


def make_run(root: Path, name: str = "moss_1m", *, step: int = 50, max_steps: int = 100,
             malformed: bool = True, samples: bool = True, checkpoints: bool = True) -> Path:
    run = root / name
    rows = [
        f"{value},{1.0 - value / 100:.4f},1e-4,0.5000,0.1000,2.000,12.50"
        for value in range(2, step + 1, 2)
    ]
    # The newest row wins over the earlier duplicate for step 50.
    rows.append(f"{step},0.5000,2e-4,0.4000,0.2000,2.000,12.50")
    if malformed:
        rows.extend(
            [
                "oops",
                f"{step + 6},0.4000,1e-4",  # truncated row
                f"{step + 8},0.4000,1e-4,0.5,0.1,2.0,12.5,extra",  # extra column
                f"{step + 10},not-a-number,1e-4,0.5,0.1,2.0,12.5",  # non-numeric value
            ]
        )
    write_metrics(run, rows)
    write_state(run, step=step, max_steps=max_steps)
    (run / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 16,
                "alpha": 32,
                "dropout": 0.0,
                "targets": ["q", "v"],
                "base_model": "models/MOSS-Image-Base",
                "revision": "abc123",
            }
        ),
        encoding="utf-8",
    )
    if samples:
        sample_dir = run / "samples" / "step_000050"
        write_png(sample_dir / "a.png")
        (sample_dir / "a.pt").write_bytes(b"latent")
        (sample_dir / "notes.txt").write_text("ignored", encoding="utf-8")
        write_png(run / "samples" / "step_000010" / "c.jpg")
    if checkpoints:
        write_checkpoint(run, 50, adapter_size=100, optimizer_size=200)
        write_checkpoint(run, 10, adapter_size=50, optimizer_size=150)
    return run


@pytest.fixture()
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    make_run(root)
    return root


@pytest.fixture()
def client_factory():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from moss_lora.dashboard import create_app

    def build(root: Path, **kwargs):
        return TestClient(create_app(root, **kwargs))

    return build


# --------------------------------------------------------------------------- discovery


def test_discovers_nested_runs_sorted_by_metrics_mtime(run_root: Path, tmp_path: Path) -> None:
    nested = run_root / "project" / "second"
    write_metrics(nested, ["1,0.9,1e-4,0.5,0.1,2.0,10.0"])
    old = run_root / "moss_1m" / "metrics.csv"
    os.utime(old, (1_000_000, 1_000_000))

    store = RunStore(run_root)
    ids = store.run_ids()
    assert ids == ["project/second", "moss_1m"]
    assert store.list_runs()[0]["id"] == "project/second"


def test_root_itself_is_a_run(tmp_path: Path) -> None:
    root = tmp_path / "moss_1m"
    write_metrics(root, ["1,0.9,1e-4,0.5,0.1,2.0,10.0"])
    write_state(root, step=1, max_steps=10)
    store = RunStore(root)
    assert store.run_ids() == ["."]
    detail = store.detail(".")
    assert detail["id"] == "."
    assert detail["name"] == "moss_1m"
    assert detail["step"] == 1


def test_empty_root(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "missing")
    assert store.run_ids() == []
    assert store.list_runs() == []


def test_rejects_path_traversal_and_escaping_symlinks(run_root: Path, tmp_path: Path) -> None:
    store = RunStore(run_root)
    for bad in ("../moss_1m", "/etc", "project/../../moss_1m", "..", "C:/x"):
        with pytest.raises(ValueError):
            store.run_dir(bad)

    outside = tmp_path / "outside"
    write_metrics(outside, ["1,0.9,1e-4,0.5,0.1,2.0,10.0"])
    os.symlink(outside, run_root / "escape")
    with pytest.raises(LookupError):
        store.run_dir("escape")


# --------------------------------------------------------------------------- parsing


def test_metrics_parsing_dedupes_and_skips_malformed(run_root: Path) -> None:
    detail = RunStore(run_root).detail("moss_1m")
    steps = [row["step"] for row in detail["metrics"]]
    assert steps == list(range(2, 51, 2))
    assert detail["counts"]["metrics"] == 25
    assert detail["metrics"][-1]["step"] == 50
    assert detail["metrics"][-1]["loss"] == pytest.approx(0.5)
    assert detail["stats"]["loss"] == pytest.approx(0.5)
    assert detail["stats"]["lr"] == pytest.approx(2e-4)
    assert detail["stats"]["peak_vram_gb"] == pytest.approx(12.5)


def test_loss_ema_follows_last_duplicate_row(tmp_path: Path) -> None:
    run = tmp_path / "ema"
    write_metrics(
        run,
        [
            "1,1.0,1e-4,0.5,0.1,2.0,10.0",
            "2,2.0,1e-4,0.5,0.1,2.0,10.0",
            "3,3.0,1e-4,0.5,0.1,2.0,10.0",
            "3,4.0,1e-4,0.5,0.1,2.0,10.0",
        ],
    )
    detail = RunStore(tmp_path).detail("ema")
    rows = detail["metrics"]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert [row["loss"] for row in rows] == [1.0, 2.0, 4.0]

    alpha = 2.0 / 21.0
    second = 1.0 + alpha * (2.0 - 1.0)
    third = second + alpha * (4.0 - second)
    assert rows[0]["loss_ema"] == pytest.approx(1.0)
    assert rows[1]["loss_ema"] == pytest.approx(second)
    assert rows[2]["loss_ema"] == pytest.approx(third)


def test_downsample_keeps_last_point_and_limit(tmp_path: Path) -> None:
    run = tmp_path / "long"
    rows = [f"{step},0.5,1e-4,0.5,0.1,2.0,10.0" for step in range(1, 5001)]
    write_metrics(run, rows)
    detail = RunStore(tmp_path).detail("long")
    points = detail["metrics"]
    assert 0 < len(points) <= 1200
    assert points[-1]["step"] == 5000
    assert detail["counts"]["metrics"] == 5000
    assert all(a["step"] < b["step"] for a, b in zip(points, points[1:]))


def test_state_and_adapter_config(run_root: Path) -> None:
    detail = RunStore(run_root).detail("moss_1m")
    assert detail["max_steps"] == 100
    assert detail["epoch"] == 2
    assert detail["batch_index"] == 10
    assert detail["adapter"]["rank"] == 16
    assert detail["adapter"]["alpha"] == 32
    assert detail["config"]["rank"] == 16
    assert detail["config"]["alpha"] == 32
    assert detail["config"]["cache_dir"] == "cache/1m"
    assert detail["config"]["batch_size"] == 1
    assert detail["config"]["base_model"] == "models/MOSS-Image-Base"
    assert detail["config"]["learning_rate"] == pytest.approx(1e-4)


def test_state_falls_back_to_checkpoint_and_tolerates_bad_json(run_root: Path) -> None:
    run = run_root / "moss_1m"
    (run / "trainer_state.json").write_text("{not json", encoding="utf-8")
    fallback = run / "checkpoints" / "step_000050" / "trainer_state.json"
    fallback.write_text(json.dumps({"step": 50, "config": {"train": {"max_steps": 80}}}), encoding="utf-8")
    detail = RunStore(run_root).detail("moss_1m")
    assert detail["max_steps"] == 80


def test_partial_root_state_merges_missing_checkpoint_fields(run_root: Path) -> None:
    run = run_root / "moss_1m"
    (run / "trainer_state.json").write_text(json.dumps({"step": 50}), encoding="utf-8")
    detail = RunStore(run_root).detail("moss_1m")
    assert detail["step"] == 50
    assert detail["max_steps"] == 100
    assert detail["config"]["max_steps"] == 100


# --------------------------------------------------------------------------- status / ETA


def test_running_status_and_eta(run_root: Path) -> None:
    detail = RunStore(run_root).detail("moss_1m")
    assert detail["status"] == "running"
    assert detail["eta_seconds"] == 100  # (100 - 50) * median(2.0)
    assert detail["eta_at"].endswith("Z")
    assert detail["stale_seconds"] is not None and detail["stale_seconds"] < 300


def test_finished_status(run_root: Path) -> None:
    run = run_root / "moss_1m"
    write_state(run, step=50, max_steps=50)
    detail = RunStore(run_root).detail("moss_1m")
    assert detail["status"] == "finished"
    assert detail["eta_seconds"] is None


def test_stale_status(run_root: Path) -> None:
    os.utime(run_root / "moss_1m" / "metrics.csv", (1_000_000, 1_000_000))
    detail = RunStore(run_root).detail("moss_1m")
    assert detail["status"] == "stale"
    assert detail["eta_seconds"] is None
    assert detail["stale_seconds"] > 300


def test_header_only_metrics_is_unknown(tmp_path: Path) -> None:
    run = tmp_path / "quiet"
    write_metrics(run, [])
    write_state(run, step=10, max_steps=100)
    detail = RunStore(tmp_path).detail("quiet")
    assert detail["status"] == "unknown"
    assert detail["metrics"] == []
    assert detail["stats"]["loss"] is None


# --------------------------------------------------------------------------- samples / checkpoints


def test_sample_groups_and_latents(run_root: Path) -> None:
    detail = RunStore(run_root).detail("moss_1m")
    groups = detail["samples"]
    assert [group["step"] for group in groups] == [50, 10]
    items = {item["name"]: item for item in groups[0]["items"]}
    assert list(items) == ["a.png", "a.pt"]
    assert items["a.png"]["type"] == "image"
    assert items["a.png"]["url"].startswith("/media?run=moss_1m&file=samples%2Fstep_000050%2Fa.png")
    assert items["a.pt"]["type"] == "latent"
    assert "url" not in items["a.pt"]
    assert detail["counts"]["samples"] == 3
    assert detail["sample_count"] == 3


def test_checkpoint_listing(run_root: Path) -> None:
    detail = RunStore(run_root).detail("moss_1m")
    entries = detail["checkpoints"]
    assert [entry["step"] for entry in entries] == [50, 10]
    assert entries[0]["adapter_bytes"] == 100
    assert entries[0]["optimizer_bytes"] == 200
    assert entries[0]["adapter_size"] == "100 B"
    assert entries[0]["has_state"] is True
    assert {item["name"] for item in entries[0]["files"]} >= {
        "adapter_model.safetensors", "optimizer.pt", "trainer_state.json"
    }


# --------------------------------------------------------------------------- HTTP API


def test_index_and_no_store_headers(client_factory, run_root: Path) -> None:
    client = client_factory(run_root)
    response = client.get("/")
    assert response.status_code == 200
    assert "MOSS-IMAGE" in response.text
    assert "loss tape" in response.text
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert client.get("/healthz").json() == {"status": "ok"}


def test_api_runs_and_detail(client_factory, run_root: Path) -> None:
    client = client_factory(run_root)
    listing = client.get("/api/runs")
    assert listing.status_code == 200
    payload = listing.json()
    assert payload["root"] == str(run_root.resolve())
    assert len(payload["runs"]) == 1
    summary = payload["runs"][0]
    assert summary["id"] == "moss_1m"
    assert summary["step"] == 50
    assert summary["max_steps"] == 100
    assert summary["status"] == "running"
    assert summary["sample_count"] == 3
    assert summary["checkpoint_count"] == 2

    detail = client.get("/api/run", params={"path": "moss_1m"})
    assert detail.status_code == 200
    body = detail.json()
    for key in (
        "config", "stats", "metrics", "samples", "checkpoints", "status", "eta_seconds",
        "updated_at", "stale_seconds", "refresh_seconds", "counts",
    ):
        assert key in body
    assert body["refresh_seconds"] == 5
    assert body["metrics"][-1]["step"] == 50
    assert body["config"]["rank"] == 16
    assert "adapter_model.safetensors" in {item["name"] for item in body["checkpoints"][0]["files"]}


def test_api_errors_are_json(client_factory, run_root: Path) -> None:
    client = client_factory(run_root)
    missing = client.get("/api/run")
    assert missing.status_code == 400
    assert "path" in missing.json()["error"]

    traversal = client.get("/api/run", params={"path": "../moss_1m"})
    assert traversal.status_code == 400
    absolute = client.get("/api/run", params={"path": "/etc"})
    assert absolute.status_code == 400

    unknown = client.get("/api/run", params={"path": "nope"})
    assert unknown.status_code == 404
    assert "not found" in unknown.json()["error"].lower()


def test_media_serves_images_only(client_factory, run_root: Path) -> None:
    client = client_factory(run_root)
    response = client.get("/media", params={"run": "moss_1m", "file": "samples/step_000050/a.png"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.content == (run_root / "moss_1m" / "samples" / "step_000050" / "a.png").read_bytes()

    for path in (
        "samples/step_000050/a.pt",
        "samples/step_000050/notes.txt",
        "../trainer_state.json",
        "/etc/passwd",
    ):
        denied = client.get("/media", params={"run": "moss_1m", "file": path})
        assert denied.status_code == 404, path

    bad_run = client.get("/media", params={"run": "../moss_1m", "file": "samples/step_000050/a.png"})
    assert bad_run.status_code in (400, 404)


def test_media_rejects_symlink_escape(client_factory, run_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(outside)
    link = run_root / "moss_1m" / "samples" / "step_000050" / "leak.png"
    os.symlink(outside, link)
    client = client_factory(run_root)
    response = client.get("/media", params={"run": "moss_1m", "file": "samples/step_000050/leak.png"})
    assert response.status_code == 404


def test_root_is_a_run_over_http(client_factory, tmp_path: Path) -> None:
    root = tmp_path / "moss_1m"
    write_metrics(root, ["1,0.9,1e-4,0.5,0.1,2.0,10.0"])
    write_state(root, step=1, max_steps=10)
    client = client_factory(root)
    listing = client.get("/api/runs").json()
    assert [run["id"] for run in listing["runs"]] == ["."]
    assert client.get("/api/run", params={"path": "."}).status_code == 200


def test_empty_root_over_http(client_factory, tmp_path: Path) -> None:
    client = client_factory(tmp_path / "nowhere")
    listing = client.get("/api/runs")
    assert listing.status_code == 200
    assert listing.json()["runs"] == []
    assert client.get("/").status_code == 200
    assert client.get("/api/run", params={"path": "."}).status_code == 404
