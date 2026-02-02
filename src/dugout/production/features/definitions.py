"""Feature column definitions and configuration.

Defines the feature columns expected by the trained LightGBM models.

Two model variants:
- BASE_FEATURES: For Captain and Transfer decisions (no cost - pure ranking)
- FREE_HIT_FEATURES: For Free Hit optimization (includes cost - budget ceiling problem)

"Cost is excluded from ranking decisions but included in Free Hit optimization 
because Free Hit is a budget-constrained selection problem where price proxies 
player ceiling."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# Base features for ranking decisions (Captain, Transfer)
# Cost excluded to avoid circular reasoning and find inefficiencies
BASE_FEATURES = [
    # Weighted performance (decay-weighted over last 5)
    "per90_wmean",
    "per90_wvar",
    # Activity (last 5)
    "mins_mean",
    "appearances",
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
    "apps_x_goals",
]

# Free Hit features (includes cost for budget-constrained optimization)
# Cost included because Free Hit maximizes ceiling under budget constraint
FREE_HIT_FEATURES = BASE_FEATURES + ["now_cost"]

# Default for backwards compatibility
FEATURE_COLUMNS = BASE_FEATURES


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    feature_cols: List[str] = field(default_factory=lambda: BASE_FEATURES.copy())
