#!/usr/bin/env python
"""Create the tiny model checkpoint and synthetic cache used by configs/tiny_cpu.yaml."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.helpers import make_tiny_cache, make_tiny_model_dir

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model_dir = ROOT / "runs" / "tiny_model"
    cache_dir = ROOT / "runs" / "tiny_cache"
    make_tiny_model_dir(model_dir)
    make_tiny_cache(cache_dir, text_dim=32, latent_channels=8)
    print(f"tiny model: {model_dir}\ntiny cache: {cache_dir}")


if __name__ == "__main__":
    main()
