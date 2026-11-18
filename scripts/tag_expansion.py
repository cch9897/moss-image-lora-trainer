#!/usr/bin/env python
"""Tag staged expansion rows with a local booru tagger (thin CLI wrapper)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_lora.tagger import main  # noqa: E402

if __name__ == "__main__":
    main()
