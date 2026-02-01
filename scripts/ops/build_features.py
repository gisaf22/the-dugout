#!/usr/bin/env python3
"""Build production feature dataset.

Production-only feature build. Research uses Stages 2-4 separately.

Usage:
    PYTHONPATH=src python scripts/ops/build_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dugout.production import Pipeline


if __name__ == "__main__":
    Pipeline.run()
