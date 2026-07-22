#!/usr/bin/env python3
"""Compatibility entrypoint for the MAPPO evaluation script."""

from __future__ import annotations

from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.MAPPO.evaluate_mappo_remote_sensing import main


if __name__ == "__main__":
    main()
