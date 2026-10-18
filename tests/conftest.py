"""Pytest fixtures: tiny transformer checkpoint + synthetic cache + trainer config."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import make_tiny_cache, make_tiny_model_dir, write_config


@pytest.fixture()
def tiny_model_dir(tmp_path: Path) -> Path:
    return make_tiny_model_dir(tmp_path / "model")


@pytest.fixture()
def tiny_cache(tmp_path: Path) -> Path:
    return make_tiny_cache(tmp_path / "cache")


@pytest.fixture()
def tiny_config(tmp_path: Path, tiny_model_dir: Path, tiny_cache: Path) -> Path:
    return write_config(
        tmp_path / "tiny.yaml",
        model={"path": str(tiny_model_dir)},
        data={"cache_dir": str(tiny_cache)},
        train={"output_dir": str(tmp_path / "run")},
    )
