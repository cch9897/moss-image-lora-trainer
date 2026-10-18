"""Dataset-side cache reading, aspect-ratio bucketing and batch construction."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import safe_open
from torch.utils.data import DataLoader, Dataset, Sampler

LATENT_KEY = "latent"


class CachedDataset(Dataset):
    """Reads one safetensors per item written by moss_lora.cache.

    __getitem__ returns the fp32/bf16 latent plus one caption variant's text
    embedding picked uniformly (the report's 1:1:1 short/medium/long sampling;
    single-variant items always pick variant 0).
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        index_path = self.cache_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Cache index not found: {index_path} (run moss-lora-cache first)")
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        self.items: list[dict[str, Any]] = self.index["items"]

    def __len__(self) -> int:
        return len(self.items)

    def item_path(self, item: dict[str, Any]) -> Path:
        return self.cache_dir / f"{item['id']}.safetensors"

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        with safe_open(str(self.item_path(item)), framework="pt", device="cpu") as handle:
            latent = handle.get_tensor(LATENT_KEY)
            variant = random.randrange(len(item["text_lengths"]))
            text = handle.get_tensor(f"text_{variant}")
        return {"latents": latent, "text": text, "id": item["id"],
                "bucket": (item["bucket_h"], item["bucket_w"])}

    def text_embedding(self, item_id: str, variant: int = 0) -> torch.Tensor:
        """Deterministic access for validation sampling (variant 0 by default)."""
        for item in self.items:
            if item["id"] == item_id:
                with safe_open(str(self.item_path(item)), framework="pt", device="cpu") as handle:
                    return handle.get_tensor(f"text_{variant}")
        raise KeyError(f"Item {item_id!r} is not in the cache index.")


class BucketBatchSampler(Sampler[list[int]]):
    """Groups dataset indices by exact (bucket_h, bucket_w), shuffles buckets and within.

    Re-seeded per epoch (seed + epoch) so runs and resumes are deterministic.
    """

    def __init__(self, buckets: list[tuple[int, int]], batch_size: int, drop_last: bool = True,
                 seed: int = 42, epoch: int = 0):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = epoch
        self.groups: dict[tuple[int, int], list[int]] = {}
        for index, bucket in enumerate(buckets):
            self.groups.setdefault(tuple(bucket), []).append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def batches_for_epoch(self, epoch: int | None = None) -> list[list[int]]:
        epoch = self.epoch if epoch is None else epoch
        rng = random.Random(self.seed + epoch)
        batches: list[list[int]] = []
        for indices in self.groups.values():
            indices = list(indices)
            rng.shuffle(indices)
            if self.drop_last:
                batches.extend(indices[i:i + self.batch_size]
                               for i in range(0, len(indices) - self.batch_size + 1, self.batch_size))
            else:
                batches.extend(indices[i:i + self.batch_size]
                               for i in range(0, len(indices), self.batch_size))
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches_for_epoch())

    def __len__(self) -> int:
        total = 0
        for indices in self.groups.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total


class _FixedBatches(Sampler[list[int]]):
    """Serves a precomputed batch list (used for exact resume fast-forwarding)."""

    def __init__(self, batches: list[list[int]]):
        self.batches = batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack same-shape latents and right-pad text embeddings to the batch max.

    The bool mask marks real tokens; the DiT drops padding internally.
    """
    shapes = {tuple(item["latents"].shape) for item in batch}
    if len(shapes) != 1:
        raise RuntimeError(f"Batch mixes latent shapes {shapes}; bucket batching must group exact shapes.")
    latents = torch.stack([item["latents"] for item in batch])
    length = max(item["text"].shape[0] for item in batch)
    dim = batch[0]["text"].shape[1]
    texts = torch.zeros(len(batch), length, dim, dtype=batch[0]["text"].dtype)
    mask = torch.zeros(len(batch), length, dtype=torch.bool)
    for row, item in enumerate(batch):
        texts[row, :item["text"].shape[0]] = item["text"]
        mask[row, :item["text"].shape[0]] = True
    return {"latents": latents, "text": texts, "mask": mask,
            "ids": [item["id"] for item in batch]}


def _seed_worker(worker_id: int) -> None:  # noqa: ARG001
    """Give every DataLoader worker an independent, deterministic RNG stream."""
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader(dataset: CachedDataset, batches: list[list[int]], *, num_workers: int = 0,
                     pin_memory: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=_FixedBatches(batches),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
        worker_init_fn=_seed_worker,
        persistent_workers=num_workers > 0,
    )
