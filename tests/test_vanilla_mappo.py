from __future__ import annotations

from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.MAPPO.test_vanilla_mappo import *  # noqa: F401,F403
