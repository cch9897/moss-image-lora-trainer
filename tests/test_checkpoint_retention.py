"""Checkpoint retention: step-window pruning, epoch-end exemptions and config parsing."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest
import yaml

from moss_lora.config import load_config
from moss_lora.trainer import prune_checkpoints


def _make_checkpoint(run: Path, step: int, *, epoch_end: bool = False,
                     trainer_state: bool = True) -> Path:
    directory = run / "checkpoints" / f"step_{step:06d}"
    directory.mkdir(parents=True)
    (directory / "adapter_model.safetensors").write_bytes(b"adapter")
    if trainer_state:
        state = {"step": step, "epoch": 1, "batch_index": 0, "epoch_end": epoch_end}
        (directory / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    return directory


def _steps(paths: Iterable[Path]) -> list[int]:
    return sorted(int(path.name.split("_", 1)[1]) for path in paths)


def _remaining_steps(run: Path) -> list[int]:
    return _steps(path for path in (run / "checkpoints").glob("step_*") if path.is_dir())


def test_prune_keeps_newest_keep_last_steps(tmp_path: Path) -> None:
    run = tmp_path / "run"
    # Creation (and therefore mtime and glob) order deliberately differs from the step order.
    for step in (8, 3, 1, 6, 4, 7, 2, 5):
        _make_checkpoint(run, step)

    removed = prune_checkpoints(run, 5)

    assert _remaining_steps(run) == [4, 5, 6, 7, 8]
    assert _steps(removed) == [1, 2, 3]


def test_epoch_end_checkpoints_survive_outside_the_window(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for step in range(1, 9):
        _make_checkpoint(run, step, epoch_end=(step == 2))

    removed = prune_checkpoints(run, 3)

    assert _remaining_steps(run) == [2, 6, 7, 8]
    assert _steps(removed) == [1, 3, 4, 5]


def test_epoch_end_checkpoint_does_not_consume_a_regular_slot(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for step in range(1, 7):
        _make_checkpoint(run, step, epoch_end=(step == 5))

    removed = prune_checkpoints(run, 3)

    # The newest 3 *regular* checkpoints survive; epoch-end step 5 is extra.
    assert _remaining_steps(run) == [3, 4, 5, 6]
    assert _steps(removed) == [1, 2]


@pytest.mark.parametrize("keep_last", [pytest.param(0, id="zero"), pytest.param(-1, id="negative")])
def test_non_positive_keep_last_prunes_nothing(tmp_path: Path, keep_last: int) -> None:
    run = tmp_path / "run"
    for step in (1, 2, 3):
        _make_checkpoint(run, step)

    assert prune_checkpoints(run, keep_last) == []
    assert _remaining_steps(run) == [1, 2, 3]


def test_incomplete_checkpoints_are_ignored_and_never_counted(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for step in range(1, 7):
        _make_checkpoint(run, step)
    # Newest directory, still being written: no trainer_state.json yet.
    in_progress = _make_checkpoint(run, 9, trainer_state=False)
    (run / "checkpoints" / "not_a_checkpoint").mkdir()

    removed = prune_checkpoints(run, 2)

    assert _remaining_steps(run) == [5, 6, 9]
    assert (in_progress / "adapter_model.safetensors").is_file()
    assert (run / "checkpoints" / "not_a_checkpoint").is_dir()
    assert _steps(removed) == [1, 2, 3, 4]


def test_load_config_reads_checkpoint_retention_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"train": {"keep_last_checkpoints": 5, "save_epoch": True}}),
                    encoding="utf-8")

    config = load_config(path)

    assert config.train.keep_last_checkpoints == 5
    assert config.train.save_epoch is True


def test_load_config_rejects_negative_keep_last(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"train": {"keep_last_checkpoints": -1}}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"keep_last_checkpoints must be >= 0"):
        load_config(path)
