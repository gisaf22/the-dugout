"""Feature column definitions and configuration.

Defines the feature columns expected by the trained LightGBM model.
These columns must be present in any prediction input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


FEATURE_COLUMNS = [
    # Weighted performance (decay-weighted over last 5)
    "per90_wmean",
    "per90_wvar",
    # Activity (last 5)
    "mins_mean",
    "appearances",
    # Current state
    "now_cost",
    # Fixture
    "is_home_next",
    # Temporal
    "games_since_first",
    # Performance per90 (last 5)
    "goals_per90",
    "assists_per90",
    "bonus_per90",
    "bps_per90",
    "ict_per90",
    "xg_per90",
    "xa_per90",
    # Interactions
    "ict_per90_x_mins",
    "xg_per90_x_apps",
]


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    feature_cols: List[str] = field(default_factory=lambda: FEATURE_COLUMNS.copy())
