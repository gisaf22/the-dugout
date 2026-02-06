"""Centralized configuration for The Dugout.

All paths, model parameters, and settings in one place.
Environment variables can override defaults.

Path Constants:
    PROJECT_ROOT - Root directory of the project
    STORAGE_DIR - Storage for database and datasets
    MODELS_DIR - Trained model artifacts
    DEFAULT_DB_PATH - SQLite database (overridable via DUGOUT_DB_PATH)
    DEFAULT_MODEL_PATH - Default LightGBM model

FPL Constants:
    FPL_BASE_URL - Official FPL API base URL
    ELEMENT_TYPE_TO_POS - Map element_type (1-4) to position codes
    FPL_POSITION_MAP - Same as above

Environment Variables:
    DUGOUT_DB_PATH - Override default database path
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root (src/dugout/production/config.py -> production -> dugout -> src -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage" / "production"
DATA_DIR = PROJECT_ROOT / "storage"  # Shared storage (for database)
MODELS_DIR = STORAGE_DIR / "models"
MODEL_DIR = MODELS_DIR  # Alias for backwards compatibility

# Database paths (database is shared, at storage/ root)
DEFAULT_DB_PATH = os.environ.get(
    "DUGOUT_DB_PATH",
    str(DATA_DIR / "fpl_2025_26.sqlite")
)

# Model artifacts
# NOTE: Legacy single-stage is production default (lower captain regret).
# Two-stage remains available for research/interpretation.
DEFAULT_MODEL_PATH = MODELS_DIR / "lightgbm_v2" / "model.joblib"

# FPL API
FPL_BASE_URL = "https://fantasy.premierleague.com/api"

def get_current_season_id() -> int:
    """Calculate season ID from current date.
    
    FPL season IDs: 2015/16=0, 2016/17=1, ..., 2024/25=9, 2025/26=10
    Season runs Aug-May: Aug-Dec uses current year, Jan-Jul uses previous year.
    """
    from datetime import datetime
    now = datetime.now()
    start_year = now.year if now.month >= 8 else now.year - 1
    return start_year - 2015

CURRENT_SEASON_ID = get_current_season_id()

# Squad constraints
BUDGET = 100.0
MAX_PLAYERS_PER_TEAM = 3
SQUAD_SIZE = 15
STARTING_XI = 11

# Position constraints
POSITION_CONSTRAINTS = {
    "GKP": {"min": 2, "max": 2, "xi_min": 1, "xi_max": 1},
    "DEF": {"min": 5, "max": 5, "xi_min": 3, "xi_max": 5},
    "MID": {"min": 5, "max": 5, "xi_min": 2, "xi_max": 5},
    "FWD": {"min": 3, "max": 3, "xi_min": 1, "xi_max": 3},
}

# Position mapping (element_type -> position code)
ELEMENT_TYPE_TO_POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POS_TO_ELEMENT_TYPE = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
FPL_POSITION_MAP = ELEMENT_TYPE_TO_POS  # Alias for convenience

# Rate limiting for API
REQUEST_DELAY = 1.0
MAX_RETRIES = 5
