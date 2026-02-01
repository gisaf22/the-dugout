#!/usr/bin/env python3
"""Automated FPL data puller.

Fetches latest data from the official FPL API and updates the local database.
Designed to run ~5 hours after each gameweek deadline.

Usage:
    python scripts/pull_fpl_data.py              # Pull latest GW
    python scripts/pull_fpl_data.py --gw 18      # Pull specific GW
    python scripts/pull_fpl_data.py --full       # Full season refresh

Environment:
    DUGOUT_DB_PATH: Path to SQLite database (default: storage/fpl_2025_26.sqlite)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dugout.production.data import FPLPuller
from dugout.production.config import DEFAULT_DB_PATH

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Pull FPL data from official API."""
    parser = argparse.ArgumentParser(description="Pull FPL data from official API")
    parser.add_argument("--gw", type=int, help="Specific gameweek to pull")
    parser.add_argument("--full", action="store_true", help="Full season refresh")
    parser.add_argument("--db", default=None, help="Database path (default: from config)")
    parser.add_argument("--force", action="store_true", help="Skip deadline check")
    args = parser.parse_args()
    
    db_path = args.db or os.environ.get("DUGOUT_DB_PATH") or DEFAULT_DB_PATH
    
    puller = FPLPuller(db_path)
    
    # Determine target GW
    target_gw = args.gw or puller.get_current_gw()
    
    if not target_gw:
        logger.error("Could not determine gameweek to pull")
        sys.exit(1)
    
    # Check if GW is finished (skip if still in progress, unless forced)
    if not args.full and not args.force and not puller.is_gw_finished(target_gw):
        logger.info(f"GW{target_gw} not finished yet. Use --force to pull anyway.")
        sys.exit(0)
    
    # Pull data: full refresh ignores gw, single GW ignores full
    if args.full:
        success = puller.update_database(full=True)
    else:
        success = puller.update_database(gw=target_gw)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
