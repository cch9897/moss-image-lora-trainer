"""Read-only live telemetry dashboard for MOSS-Image LoRA runs.

The trainer writes everything this module reads: ``metrics.csv``, sample images under
``samples/step_*``, checkpoints under ``checkpoints/step_*``, ``adapter_config.json`` and
``trainer_state.json``. Nothing here mutates a run, and weights are never served.

FastAPI/uvicorn are optional dependencies (``pip install -e '.[dashboard]'``); importing this
module works without them, and only :func:`create_app` / :func:`main` need them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import os
import statistics
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlencode

METRICS_NAME = "metrics.csv"
TRAINER_STATE_NAME = "trainer_state.json"
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
OPTIMIZER_NAME = "optimizer.pt"
SAMPLES_DIR = "samples"
CHECKPOINTS_DIR = "checkpoints"

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
METRIC_FIELDS = ("loss", "lr", "sigma", "grad_norm", "sec_per_step", "peak_vram_gb")
EMA_WINDOW = 20
MAX_METRIC_POINTS = 1200
DEFAULT_REFRESH_SECONDS = 5.0
DEFAULT_STALE_AFTER = 300.0
MAX_SCAN_DEPTH = 4
MAX_JSON_BYTES = 2 * 1024 * 1024

SKIP_DIRS = frozenset(
    {".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules", SAMPLES_DIR, CHECKPOINTS_DIR}
)
NO_STORE = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache", "Expires": "0"}
DASHBOARD_EXTRA_HINT = (
    "The dashboard needs optional dependencies. Install them with: "
    "pip install -e '.[dashboard]'"
)
DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")


# --------------------------------------------------------------------------- small helpers


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _resolve(path: Path) -> Path | None:
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def _contained_file(path: Path, base: Path) -> Path | None:
    """Resolve ``path`` and return it only when it is a regular file inside ``base``."""
    resolved = _resolve(path)
    if resolved is None or not resolved.is_file() or not _is_within(resolved, base):
        return None
    return resolved


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _now_iso(after_seconds: float | None = None) -> str | None:
    if after_seconds is None:
        return None
    return _iso(time.time() + after_seconds)


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_int(value: Any) -> int | None:
    number = _int_value(value)
    return number if number and number > 0 else None


def _normalize_run_id(run_id: str) -> str:
    """Validate a client-supplied run id and return its canonical POSIX form."""
    if not isinstance(run_id, str):
        raise ValueError("Run id must be a string.")
    raw = run_id.strip()
    if not raw:
        raise ValueError("Run id is empty.")
    if "\x00" in raw or "\\" in raw:
        raise ValueError("Run id contains characters that are not allowed.")
    if raw in (".", "./"):
        return "."
    path = PurePosixPath(raw)
    if path.is_absolute():
        raise ValueError("Run id must be relative.")
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts:
        return "."
    if any(part == ".." for part in parts):
        raise ValueError("Run id must not contain '..'.")
    if parts[0].endswith(":"):
        raise ValueError("Run id must not look like a drive path.")
    return "/".join(parts)


def _step_from_name(name: str) -> int | None:
    if not name.startswith("step_"):
        return None
    return _int_value(name[5:])


def _ema(rows: list[dict[str, Any]], window: int = EMA_WINDOW) -> None:
    alpha = 2.0 / (window + 1.0)
    current: float | None = None
    for row in rows:
        loss = row.get("loss")
        if loss is not None:
            current = loss if current is None else current + alpha * (loss - current)
        row["loss_ema"] = current


def _downsample(rows: list[dict[str, Any]], limit: int = MAX_METRIC_POINTS) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    stride = math.ceil(len(rows) / limit)
    sampled = rows[::stride]
    if sampled and sampled[-1]["step"] != rows[-1]["step"]:
        if len(sampled) >= limit:
            sampled = sampled[: limit - 1]
        sampled.append(rows[-1])
    return sampled


def _parse_metrics(path: Path) -> list[dict[str, Any]]:
    """Parse an append-only metrics CSV, tolerating partial writes.

    Rows with extra/missing columns, an invalid step, or a non-finite value are skipped. Later
    duplicates of a step replace earlier ones, and the result is sorted by step.
    """
    rows_by_step: dict[int, dict[str, Any]] = {}
    try:
        handle = path.open("r", encoding="utf-8-sig", errors="replace", newline="")
    except OSError:
        return []
    with handle:
        try:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            names = [str(name).strip() for name in reader.fieldnames]
            if "step" not in names:
                return []
            reader.fieldnames = names
            for raw in reader:
                if None in raw:  # extra columns
                    continue
                if any(raw.get(name) is None for name in names):  # truncated row
                    continue
                step = _int_value(raw.get("step"))
                if step is None:
                    continue
                row: dict[str, Any] = {"step": step}
                malformed = False
                for field in METRIC_FIELDS:
                    text = raw.get(field)
                    value = _float_value(text)
                    if value is None and text is not None and str(text).strip():
                        malformed = True
                        break
                    row[field] = value
                if malformed:
                    continue
                rows_by_step[step] = row
        except csv.Error:
            return [rows_by_step[step] for step in sorted(rows_by_step)]
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def _read_json(path: Path, base: Path) -> dict[str, Any] | None:
    resolved = _contained_file(path, base)
    if resolved is None:
        return None
    try:
        if resolved.stat().st_size > MAX_JSON_BYTES:
            return None
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _last_value(rows: list[dict[str, Any]], field: str) -> float | None:
    for row in reversed(rows):
        value = row.get(field)
        if value is not None:
            return value
    return None


def _human_size(size: int | None) -> str | None:
    if size is None:
        return None
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return None


# --------------------------------------------------------------------------- run store


class RunStore:
    """Read-only access to one or more run directories under ``root``."""

    def __init__(self, root: str | os.PathLike[str], *, stale_after: float = DEFAULT_STALE_AFTER,
                 max_depth: int = MAX_SCAN_DEPTH) -> None:
        self.root = Path(root).expanduser()
        resolved = _resolve(self.root)
        self.root = resolved if resolved is not None else self.root.absolute()
        self.stale_after = max(0.0, float(stale_after))
        self.max_depth = max(1, int(max_depth))

    # -- discovery ---------------------------------------------------------

    def _candidate_dirs(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        if _contained_file(self.root / METRICS_NAME, self.root) is not None:
            return [self.root]
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            current = Path(dirpath)
            rel = current.relative_to(self.root)
            depth = 0 if rel == Path(".") else len(rel.parts)
            keep = []
            for name in dirnames:
                if name.startswith(".") or name in SKIP_DIRS:
                    continue
                if os.path.islink(current / name):
                    continue
                if depth + 1 > self.max_depth:
                    continue
                keep.append(name)
            dirnames[:] = sorted(keep)
            if METRICS_NAME in filenames:
                resolved = _resolve(current)
                if resolved is not None and _is_within(resolved, self.root):
                    found.append(resolved)
        return found

    def run_ids(self) -> list[str]:
        return [entry["id"] for entry in self._discover()]

    def _discover(self) -> list[dict[str, Any]]:
        entries = []
        for directory in self._candidate_dirs():
            metrics = _contained_file(directory / METRICS_NAME, directory)
            if metrics is None:
                continue
            rel = directory.relative_to(self.root)
            run_id = "." if rel == Path(".") else rel.as_posix()
            try:
                mtime = metrics.stat().st_mtime
            except OSError:
                continue
            entries.append({"id": run_id, "mtime": mtime})
        entries.sort(key=lambda item: (-item["mtime"], item["id"]))
        return entries

    def run_dir(self, run_id: str) -> Path:
        normalized = _normalize_run_id(run_id)
        candidate = self.root if normalized == "." else self.root.joinpath(*normalized.split("/"))
        resolved = _resolve(candidate)
        if resolved is None or not resolved.is_dir() or not _is_within(resolved, self.root):
            raise LookupError(f"Run not found: {run_id}")
        return resolved

    # -- run files ---------------------------------------------------------

    def _metrics(self, run_dir: Path) -> tuple[list[dict[str, Any]], Path | None]:
        metrics = _contained_file(run_dir / METRICS_NAME, run_dir)
        if metrics is None:
            return [], None
        return _parse_metrics(metrics), metrics

    def _checkpoint_dirs(self, run_dir: Path) -> list[tuple[int, Path]]:
        root = _resolve(run_dir / CHECKPOINTS_DIR)
        if root is None or not root.is_dir() or not _is_within(root, run_dir):
            return []
        entries = []
        for child in root.iterdir():
            step = _step_from_name(child.name)
            if step is None:
                continue
            resolved = _resolve(child)
            if resolved is None or not resolved.is_dir() or not _is_within(resolved, run_dir):
                continue
            entries.append((step, resolved))
        entries.sort(key=lambda item: item[0], reverse=True)
        return entries

    @staticmethod
    def _state_payload(payload: dict[str, Any], source: str) -> dict[str, Any]:
        config = payload.get("config")
        config = config if isinstance(config, dict) else {}
        train = config.get("train") if isinstance(config.get("train"), dict) else {}
        return {
            "step": _int_value(payload.get("step")),
            "epoch": _int_value(payload.get("epoch")),
            "batch_index": _int_value(payload.get("batch_index")),
            "max_steps": _positive_int(train.get("max_steps")),
            "config": config,
            "source": source,
        }

    def _state(self, run_dir: Path, checkpoints: list[tuple[int, Path]]) -> dict[str, Any]:
        """Read trainer_state.json from the run root, else from the newest checkpoint.

        A root-level state file wins, but missing fields are filled from the newest checkpoint
        so a partially written root file does not hide max_steps or the config.
        """
        root_payload = _read_json(run_dir / TRAINER_STATE_NAME, run_dir)
        if root_payload is not None:
            state = self._state_payload(root_payload, "run")
            if state["max_steps"] is None or not state["config"]:
                for _, directory in checkpoints:
                    fallback_payload = _read_json(directory / TRAINER_STATE_NAME, run_dir)
                    if fallback_payload is None:
                        continue
                    fallback = self._state_payload(fallback_payload, "checkpoint")
                    for key in ("step", "epoch", "batch_index"):
                        if state[key] is None:
                            state[key] = fallback[key]
                    if state["max_steps"] is None:
                        state["max_steps"] = fallback["max_steps"]
                    if not state["config"]:
                        state["config"] = fallback["config"]
                    break
            return state
        for _, directory in checkpoints:
            payload = _read_json(directory / TRAINER_STATE_NAME, run_dir)
            if payload is not None:
                return self._state_payload(payload, "checkpoint")
        return {"step": None, "epoch": None, "batch_index": None, "max_steps": None,
                "config": {}, "source": None}

    def _adapter(self, run_dir: Path, checkpoints: list[tuple[int, Path]]) -> dict[str, Any]:
        payload = None
        for directory in [run_dir, *(directory for _, directory in checkpoints)]:
            payload = _read_json(directory / ADAPTER_CONFIG_NAME, run_dir)
            if payload is not None:
                break
        if payload is None:
            return {}
        rank = _positive_int(payload.get("r"))
        alpha = _float_value(payload.get("alpha"))
        targets = payload.get("targets")
        return {
            "rank": rank,
            "alpha": alpha if alpha is not None else rank,
            "dropout": _float_value(payload.get("dropout")),
            "targets": [str(value) for value in targets] if isinstance(targets, list) else [],
            "base_model": payload.get("base_model") if isinstance(payload.get("base_model"), str) else None,
            "revision": payload.get("revision") if isinstance(payload.get("revision"), str) else None,
            "text_projection": bool(payload.get("text_projection", False)),
            "final_linear": bool(payload.get("final_linear", False)),
        }

    def _samples(self, run_dir: Path, run_id: str) -> list[dict[str, Any]]:
        root = _resolve(run_dir / SAMPLES_DIR)
        if root is None or not root.is_dir() or not _is_within(root, run_dir):
            return []
        groups = []
        for step_dir in root.iterdir():
            step = _step_from_name(step_dir.name)
            if step is None or step_dir.is_symlink():
                continue
            resolved_dir = _resolve(step_dir)
            if resolved_dir is None or not resolved_dir.is_dir() or not _is_within(resolved_dir, run_dir):
                continue
            items = []
            try:
                children = sorted(resolved_dir.iterdir(), key=lambda child: child.name)
            except OSError:
                continue
            for child in children:
                suffix = child.suffix.lower()
                if suffix not in IMAGE_SUFFIXES and suffix != ".pt":
                    continue
                file_path = _contained_file(child, run_dir)
                if file_path is None:
                    continue
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                item = {
                    "name": child.name,
                    "type": "image" if suffix in IMAGE_SUFFIXES else "latent",
                    "size": stat.st_size,
                    "mtime": _iso(stat.st_mtime),
                }
                if item["type"] == "image":
                    relative = file_path.relative_to(run_dir).as_posix()
                    item["url"] = "/media?" + urlencode({"run": run_id, "file": relative})
                items.append(item)
            if items:
                groups.append({"step": step, "items": items})
        groups.sort(key=lambda group: group["step"], reverse=True)
        return groups

    def _checkpoints(self, checkpoints: list[tuple[int, Path]]) -> list[dict[str, Any]]:
        entries = []
        for step, directory in checkpoints:
            files = []
            try:
                children = sorted(directory.iterdir(), key=lambda child: child.name)
            except OSError:
                continue
            for child in children:
                if child.is_symlink() or not child.is_file():
                    continue
                try:
                    stat = child.stat()
                except OSError:
                    continue
                files.append({"name": child.name, "size": stat.st_size, "mtime": _iso(stat.st_mtime)})
            by_name = {entry["name"]: entry["size"] for entry in files}
            directory_mtime = directory.stat().st_mtime
            entries.append(
                {
                    "step": step,
                    "adapter_bytes": by_name.get(ADAPTER_WEIGHTS_NAME),
                    "optimizer_bytes": by_name.get(OPTIMIZER_NAME),
                    "adapter_size": _human_size(by_name.get(ADAPTER_WEIGHTS_NAME)),
                    "optimizer_size": _human_size(by_name.get(OPTIMIZER_NAME)),
                    "file_count": len(files),
                    "files": files,
                    "mtime": _iso(directory_mtime),
                    "has_state": TRAINER_STATE_NAME in by_name,
                }
            )
        return entries

    # -- projections -------------------------------------------------------

    def _project(self, run_id: str) -> dict[str, Any]:
        directory = self.run_dir(run_id)
        rows, metrics_path = self._metrics(directory)
        _ema(rows)
        checkpoints = self._checkpoint_dirs(directory)
        state = self._state(directory, checkpoints)
        adapter = self._adapter(directory, checkpoints)
        samples = self._samples(directory, run_id)
        checkpoint_entries = self._checkpoints(checkpoints)

        step_values = [
            value for value in (rows[-1]["step"] if rows else None, state["step"]) if value is not None
        ]
        step = max(step_values) if step_values else None
        max_steps = state["max_steps"]
        mtime = metrics_path.stat().st_mtime if metrics_path is not None else None
        age = max(0.0, time.time() - mtime) if mtime is not None else None

        if not rows:
            status = "unknown"
        elif max_steps is not None and step is not None and step >= max_steps:
            status = "finished"
        elif age is not None and age <= self.stale_after:
            status = "running"
        else:
            status = "stale"

        stats = {field: _last_value(rows, field) for field in METRIC_FIELDS}
        stats["loss_ema"] = _last_value(rows, "loss_ema")
        peaks = [row["peak_vram_gb"] for row in rows if row.get("peak_vram_gb") is not None]
        stats["peak_vram_gb"] = max(peaks) if peaks else stats["peak_vram_gb"]
        latest_loss = stats["loss"]
        latest_lr = stats["lr"]

        eta_seconds = None
        if status == "running" and max_steps is not None and step is not None and step < max_steps:
            recent = [row["sec_per_step"] for row in rows[-EMA_WINDOW:]
                      if row.get("sec_per_step") is not None and row["sec_per_step"] > 0]
            if recent:
                eta_seconds = int(round((max_steps - step) * statistics.median(recent)))

        return {
            "id": run_id,
            "name": (self.root.name or str(self.root)) if run_id == "." else PurePosixPath(run_id).name,
            "directory": directory,
            "rows": rows,
            "metrics_path": metrics_path,
            "mtime": mtime,
            "age": age,
            "status": status,
            "step": step,
            "max_steps": max_steps,
            "state": state,
            "adapter": adapter,
            "stats": stats,
            "latest_loss": latest_loss,
            "latest_lr": latest_lr,
            "eta_seconds": eta_seconds,
            "samples": samples,
            "sample_count": sum(len(group["items"]) for group in samples),
            "checkpoints": checkpoint_entries,
            "checkpoint_count": len(checkpoint_entries),
        }

    def list_runs(self) -> list[dict[str, Any]]:
        summaries = []
        for entry in self._discover():
            try:
                projection = self._project(entry["id"])
            except LookupError:
                continue
            summaries.append(
                {
                    "id": projection["id"],
                    "name": projection["name"],
                    "step": projection["step"],
                    "max_steps": projection["max_steps"],
                    "latest_loss": projection["latest_loss"],
                    "latest_lr": projection["latest_lr"],
                    "updated_at": _iso(projection["mtime"]),
                    "status": projection["status"],
                    "sample_count": projection["sample_count"],
                    "checkpoint_count": projection["checkpoint_count"],
                }
            )
        return summaries

    def detail(self, run_id: str, *, refresh_seconds: float = DEFAULT_REFRESH_SECONDS) -> dict[str, Any]:
        projection = self._project(run_id)
        state = projection["state"]
        adapter = projection["adapter"]
        config = state["config"]
        train = config.get("train") if isinstance(config.get("train"), dict) else {}
        data = config.get("data") if isinstance(config.get("data"), dict) else {}
        model = config.get("model") if isinstance(config.get("model"), dict) else {}
        lora = config.get("lora") if isinstance(config.get("lora"), dict) else {}
        logging = config.get("logging") if isinstance(config.get("logging"), dict) else {}

        rank = adapter.get("rank") if adapter.get("rank") is not None else _positive_int(lora.get("rank"))
        alpha = adapter.get("alpha")
        if alpha is None:
            alpha = _float_value(lora.get("alpha"))
        if alpha is None and rank is not None:
            alpha = float(rank)

        summary_config = {
            "base_model": adapter.get("base_model") or model.get("path"),
            "cache_dir": data.get("cache_dir"),
            "rank": rank,
            "alpha": alpha,
            "dtype": model.get("dtype"),
            "learning_rate": _float_value(train.get("learning_rate")),
            "batch_size": _positive_int(data.get("batch_size")),
            "gradient_accumulation_steps": _positive_int(train.get("gradient_accumulation_steps")),
            "max_steps": projection["max_steps"],
            "warmup_steps": _int_value(train.get("warmup_steps")),
            "lr_scheduler": train.get("lr_scheduler"),
            "seed": _int_value(train.get("seed")),
            "gradient_checkpointing": train.get("gradient_checkpointing"),
            "compile": train.get("compile"),
            "ema_decay": _float_value(train.get("ema_decay")),
            "save_every": _positive_int(logging.get("save_every")),
            "log_every": _positive_int(logging.get("log_every")),
            "sample_every": _int_value(logging.get("sample_every")),
            "sample_steps": _positive_int(logging.get("sample_steps")),
            "targets": adapter.get("targets") or ([str(value) for value in lora.get("targets", [])]
                                                   if isinstance(lora.get("targets"), list) else []),
        }

        metrics = _downsample(projection["rows"])
        return {
            "id": projection["id"],
            "name": projection["name"],
            "status": projection["status"],
            "step": projection["step"],
            "max_steps": projection["max_steps"],
            "epoch": state["epoch"],
            "batch_index": state["batch_index"],
            "updated_at": _iso(projection["mtime"]),
            "stale_seconds": round(projection["age"], 1) if projection["age"] is not None else None,
            "refresh_seconds": refresh_seconds,
            "eta_seconds": projection["eta_seconds"],
            "eta_at": _now_iso(projection["eta_seconds"]),
            "stats": projection["stats"],
            "config": summary_config,
            "adapter": adapter,
            "metrics": metrics,
            "samples": projection["samples"],
            "checkpoints": projection["checkpoints"],
            "sample_count": projection["sample_count"],
            "checkpoint_count": projection["checkpoint_count"],
            "counts": {
                "metrics": len(projection["rows"]),
                "samples": projection["sample_count"],
                "checkpoints": projection["checkpoint_count"],
            },
        }


# --------------------------------------------------------------------------- FastAPI app


def create_app(root: str | os.PathLike[str], refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
               stale_after: float = DEFAULT_STALE_AFTER) -> Any:
    """Build the read-only FastAPI application for ``root``."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
        from starlette.exceptions import HTTPException as StarletteHTTPException
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(f"{DASHBOARD_EXTRA_HINT} ({exc})") from exc

    refresh_seconds = max(1.0, float(refresh_seconds))
    store = RunStore(root, stale_after=stale_after)
    app = FastAPI(
        title="MOSS-Image LoRA telemetry",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store = store

    @app.middleware("http")
    async def _no_store(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(NO_STORE)
        return response

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse({"error": "Invalid request parameters."}, status_code=400)

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        try:
            return HTMLResponse(DASHBOARD_HTML.read_text(encoding="utf-8"))
        except OSError as exc:
            raise HTTPException(500, f"dashboard.html could not be read: {exc}") from exc

    @app.get("/api/runs")
    def api_runs() -> dict[str, Any]:
        return {"root": str(store.root), "runs": store.list_runs()}

    @app.get("/api/run")
    def api_run(path: str | None = None) -> dict[str, Any]:
        if not path:
            raise HTTPException(400, "Query parameter 'path' is required.")
        try:
            return store.detail(path, refresh_seconds=refresh_seconds)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/media")
    def media(run: str | None = None, file: str | None = None) -> Any:
        if not run or not file:
            raise HTTPException(400, "Query parameters 'run' and 'file' are required.")
        try:
            run_directory = store.run_dir(run)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        if "\x00" in file or "\\" in file:
            raise HTTPException(404, "Media not found.")
        relative = PurePosixPath(file)
        if relative.is_absolute() or any(part == ".." for part in relative.parts):
            raise HTTPException(404, "Media not found.")
        if relative.suffix.lower() not in IMAGE_SUFFIXES:
            raise HTTPException(404, "Media not found.")
        candidate = run_directory.joinpath(*relative.parts)
        resolved = _resolve(candidate)
        if resolved is None or not resolved.is_file() or not _is_within(resolved, run_directory):
            raise HTTPException(404, "Media not found.")
        media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        return FileResponse(resolved, media_type=media_type)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moss-lora-dashboard",
        description="Read-only live telemetry for MOSS-Image LoRA runs.",
    )
    parser.add_argument("--root", default="runs", help="Directory containing run folders (default: runs)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--refresh", type=float, default=DEFAULT_REFRESH_SECONDS,
                        help="Browser refresh interval in seconds (default: 5)")
    parser.add_argument("--stale-after", type=float, default=DEFAULT_STALE_AFTER,
                        help="Seconds without metrics before a run is stale (default: 300)")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in a browser")
    parser.add_argument("--run", default=None, help="Run id to select on load")
    args = parser.parse_args(argv)

    try:
        app = create_app(args.root, refresh_seconds=args.refresh, stale_after=args.stale_after)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        print(f"error: {DASHBOARD_EXTRA_HINT} ({exc})", file=sys.stderr)
        return 1

    store: RunStore = app.state.store
    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{display_host}:{args.port}/"
    if args.run:
        if args.run not in store.run_ids():
            print(f"warning: run {args.run!r} is not under {store.root} yet", file=sys.stderr)
        url += "?" + urlencode({"run": args.run})

    print(f"moss-lora-dashboard: watching {store.root}")
    if not store.root.is_dir():
        print("  no run directory yet; the dashboard will show the start command")
    print(f"  {url}")
    print(f"  refresh {args.refresh:g}s · stale after {args.stale_after:g}s · Ctrl+C to stop")

    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
