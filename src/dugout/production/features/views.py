"""Decision-specific feature views.

Each decision uses a specific feature set. Models import only their view.
No feature switches inside predict(). No leakage across decisions.

See docs/research/decision_specific_modeling.md for ablation evidence.
"""

from __future__ import annotations

from typing import List

# =============================================================================
# Base Features (shared foundation)
# =============================================================================

_BASE_FEATURES: List[str] = [
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


# =============================================================================
# Decision-Specific Feature Views
# =============================================================================

# Transfer: Baseline features, no cost, no defensive
# Ranking players for transfer-in, cost irrelevant for ranking
TRANSFER_FEATURES: List[str] = _BASE_FEATURES.copy()

# Captain: Baseline + position-conditional defensive features
# xgc_per90 and clean_sheet_rate are zeroed for MID/FWD (positions 3,4)
# Ablation: -1.85 pts/GW regret vs baseline
CAPTAIN_FEATURES: List[str] = _BASE_FEATURES + [
    "xgc_per90",          # Expected goals conceded per 90 (lower = better defense)
    "clean_sheet_rate",   # Clean sheet rate over last 5 GWs
]

# Free Hit: Baseline + cost (budget-constrained optimization)
# Cost included because LP optimizer needs to correlate points with price
FREE_HIT_FEATURES: List[str] = _BASE_FEATURES + [
    "now_cost",           # Player price in millions
]


# =============================================================================
# Position Constants (for defensive feature conditioning)
# =============================================================================

# Positions where defensive features are applied (zeroed for others)
# 1=GKP, 2=DEF, 3=MID, 4=FWD
DEFENSIVE_POSITIONS: List[int] = [1, 2]  # GKP, DEF


# =============================================================================
# Feature View Registry
# =============================================================================

def get_features_for_decision(decision: str) -> List[str]:
    """Get feature list for a specific decision.
    
    Args:
        decision: One of "captain", "transfer", "free_hit"
        
    Returns:
        List of feature column names
        
    Raises:
        ValueError: If decision is unknown
    """
    views = {
        "captain": CAPTAIN_FEATURES,
        "transfer": TRANSFER_FEATURES,
        "free_hit": FREE_HIT_FEATURES,
    }
    
    if decision not in views:
        raise ValueError(f"Unknown decision: {decision}. Must be one of {list(views.keys())}")
    
    return views[decision].copy()


__all__ = [
    "CAPTAIN_FEATURES",
    "TRANSFER_FEATURES",
    "FREE_HIT_FEATURES",
    "DEFENSIVE_POSITIONS",
    "get_features_for_decision",
]
