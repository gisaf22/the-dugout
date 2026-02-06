"""Feature column definitions and configuration.

Defines the feature columns expected by the trained LightGBM models.

Three model variants:
- BASE_FEATURES: For Transfer decisions and default ranking (no cost)
- CAPTAIN_FEATURES: For Captain decision (position-conditional defensive features)
- FREE_HIT_FEATURES: For Free Hit optimization (includes cost - budget ceiling problem)

"Cost is excluded from ranking decisions but included in Free Hit optimization 
because Free Hit is a budget-constrained selection problem where price proxies 
player ceiling."

"Captain uses position-conditional defensive features (xgc_per90, clean_sheet_rate)
applied only to DEF/GKP. Research ablation showed -1.85 pts/GW regret improvement.
See docs/research/decision_specific_modeling.md for evidence."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# Base features for Transfer decisions (no cost, no defensive features)
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

# Captain features (position-conditional defensive features)
# Defensive features (xgc_per90, clean_sheet_rate) are zeroed for MID/FWD
# Applied only to DEF/GKP - reduces captain regret by 1.85 pts/GW
# See docs/research/decision_specific_modeling.md for ablation evidence
CAPTAIN_FEATURES = BASE_FEATURES + [
    "xgc_per90",          # Expected goals conceded per 90 (lower = better defense)
    "clean_sheet_rate",   # Clean sheet rate over last 5 GWs
]

# Positions where defensive features are applied (zeroed for others)
# 1=GKP, 2=DEF, 3=MID, 4=FWD
DEFENSIVE_POSITIONS = [1, 2]  # GKP, DEF

# Default for backwards compatibility
FEATURE_COLUMNS = BASE_FEATURES


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    feature_cols: List[str] = field(default_factory=lambda: BASE_FEATURES.copy())
