"""Feature column definitions and configuration.

Defines the feature columns expected by the trained LightGBM model.
These columns must be present in any prediction input.

Constants:
    FEATURE_COLUMNS - 23 features used by the model
    
Classes:
    FeatureConfig - Dataclass for configuring feature computation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# Core feature columns used by the model (last 5 games, matches FeatureBuilder output)
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
    "completed_60_count",
    "minutes_fraction",
    # Detailed stats (sums/means over last 5)
    "goals_sum",
    "assists_sum",
    "bonus_sum",
    "bps_mean",
    "creativity_mean",
    "threat_mean",
    "influence_mean",
    "ict_mean",
    # xG/xA (sums over last 5)
    "xg_sum",
    "xa_sum",
    # Minutes risk (last 5, lagged) - replaces is_inactive
    "start_rate_5",          # proportion of games started
    "mins_std_5",            # volatility in minutes
    "mins_below_60_rate_5",  # proportion of games with < 60 mins
    # Tail risk / upside
    "haul_rate_5",           # proportion of 10+ point games (right-tail frequency)
    # Interactions
    "threat_x_mins",
    "ict_x_home",
    "xg_x_apps",
]


@dataclass
class FeatureConfig:
    """Configuration for feature engineering.
    
    Attributes:
        feature_cols: List of feature column names for the model.
    
    Note: Decay weights for per90 calculations are fixed in FeatureBuilder.DECAY_WEIGHTS
        (5 gameweeks, 0.7 decay factor). Change there if needed.
    """
    
    feature_cols: List[str] = field(default_factory=lambda: FEATURE_COLUMNS.copy())
