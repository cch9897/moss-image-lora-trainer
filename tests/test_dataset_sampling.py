"""Weighted variant sampling, renormalization and backward compatibility."""

from __future__ import annotations

import random

import pytest
import torch

from moss_lora.cache import normalize_variant_weights
from moss_lora.config import DataConfig
from moss_lora.dataset import CachedDataset, collate_batch, item_variant_metadata, pick_variant

KINDS = ["original", "long", "medium", "short", "layout"]


def item(kinds=None, weights=None, lengths=None) -> dict:
    kinds = kinds or KINDS
    return {
        "id": "item_00000",
        "text_lengths": lengths or [len(kinds)] * len(kinds),
        "variant_kinds": kinds,
        **({"variant_weights": weights if weights is not None
           else normalize_variant_weights(kinds)}),
    }


def distribution(item_dict: dict, draws: int = 100_000) -> dict[str, float]:
    random.seed(1234)
    counts: dict[str, int] = {}
    for _ in range(draws):
        _, kind = pick_variant(item_dict)
        counts[kind] = counts.get(kind, 0) + 1
    return {kind: count / draws for kind, count in counts.items()}


def test_weighted_distribution_within_one_percent():
    proportions = distribution(item())
    expected = {"original": 0.16, "long": 0.64 / 3, "medium": 0.64 / 3,
                "short": 0.64 / 3, "layout": 0.20}
    for kind, value in expected.items():
        assert proportions[kind] == pytest.approx(value, abs=0.01), (kind, proportions)


def test_missing_variant_renormalizes():
    missing_layout = item(kinds=["original", "long", "medium", "short"])
    weights = missing_layout["variant_weights"]
    assert weights == pytest.approx([0.2, 0.80 / 3, 0.80 / 3, 0.80 / 3])
    proportions = distribution(missing_layout, draws=50_000)
    assert proportions["original"] == pytest.approx(0.20, abs=0.01)
    assert proportions["long"] == pytest.approx(0.80 / 3, abs=0.01)

    only_two = item(kinds=["long", "short"])
    assert only_two["variant_weights"] == pytest.approx([0.5, 0.5])
    assert distribution(only_two, 20_000)["long"] == pytest.approx(0.5, abs=0.02)


def test_seeded_sampling_is_deterministic():
    data = item()
    random.seed(99)
    first = [pick_variant(data)[0] for _ in range(50)]
    random.seed(99)
    second = [pick_variant(data)[0] for _ in range(50)]
    assert first == second


def test_old_index_without_weights_samples_uniformly():
    data = {"id": "old", "text_lengths": [4, 5, 6]}
    assert item_variant_metadata(data) == (["caption", "caption", "caption"], None)
    random.seed(7)
    indices = [pick_variant(data)[0] for _ in range(30_000)]
    assert {pick_variant(data)[1] for _ in range(5)} == {"caption"}
    for index in range(3):
        assert indices.count(index) / len(indices) == pytest.approx(1 / 3, abs=0.02)


def test_override_and_zero_weight_fallback():
    data = item()
    assert pick_variant(data, {"layout": 100.0})[1] == "layout"
    uniform = item()
    random.seed(3)
    indices = [pick_variant(uniform, {"nope": 1.0})[0] for _ in range(1000)]
    assert set(indices) == set(range(5))  # all-zero override -> uniform


def test_invalid_index_metadata_is_rejected():
    with pytest.raises(ValueError, match="variant_kinds"):
        item_variant_metadata({"id": "x", "text_lengths": [1, 2],
                               "variant_kinds": ["long"]})
    with pytest.raises(ValueError, match="sum"):
        item_variant_metadata({"id": "x", "text_lengths": [1, 2],
                               "variant_kinds": ["long", "short"],
                               "variant_weights": [0.2, 0.2]})
    with pytest.raises(ValueError, match="caption_sha256"):
        item_variant_metadata({"id": "x", "text_lengths": [1, 2],
                               "caption_sha256": ["a"]})


def test_dataset_returns_variant_kind(tiny_cache):
    dataset = CachedDataset(tiny_cache)
    sample = dataset[0]
    assert "variant_kind" in sample
    assert sample["variant_kind"] in set(dataset.items[0].get("variant_kinds", ["caption"]))
    dataset_with_override = CachedDataset(tiny_cache, variant_weights={"long": 1.0})
    kinds = {dataset_with_override[index]["variant_kind"] for index in range(len(dataset))}
    assert kinds <= {"long"} or "long" in kinds


def test_collate_includes_variant_kinds():
    batch = [
        {"latents": torch.zeros(8, 4, 4), "text": torch.ones(3, 5), "id": "a",
         "variant_kind": "long"},
        {"latents": torch.zeros(8, 4, 4), "text": torch.ones(2, 5), "id": "b"},
    ]
    collated = collate_batch(batch)
    assert collated["variant_kinds"] == ["long", "caption"]


def test_dataconfig_resolved_weights_and_validation():
    config = DataConfig(layout_weight=0.20, original_weight=0.20)
    resolved = config.resolved_variant_weights()
    assert resolved == pytest.approx({"original": 0.16, "long": 0.64 / 3,
                                      "medium": 0.64 / 3, "short": 0.64 / 3,
                                      "layout": 0.20})
    assert sum(resolved.values()) == pytest.approx(1.0)
    assert DataConfig().resolved_variant_weights() == {}
    assert DataConfig(variant_weights={"long": 1.0}).resolved_variant_weights() == {"long": 1.0}
    with pytest.raises(ValueError):
        DataConfig(layout_weight=0.5)
    with pytest.raises(ValueError):
        DataConfig(variant_weights={"long": -1.0})
