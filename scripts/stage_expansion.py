#!/usr/bin/env python
"""Stage the artist-directory expansion pool (thin CLI over moss_lora.staging)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/stage_expansion.py` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_lora.staging import main  # noqa: E402

if __name__ == "__main__":
    main()
