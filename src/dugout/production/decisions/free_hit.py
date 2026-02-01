"""Free Hit optimization function.

Decision Rule (Frozen): LP maximize Σ(predicted_points)
Subject to budget, position, and team constraints.

CONTRACT ENFORCEMENT:
    - Any modification requires updating DECISION_CONTRACT_LAYER.md
    - Any modification requires re-running research pipeline
    
GUARDRAIL: Must use predict_points() from dugout.production.models.predict.
           Direct model loading is FORBIDDEN.
"""

import pandas as pd
from typing import Tuple, Optional, Any

from dugout.production.data.reader import DataReader
from dugout.production.features.builder import FeatureBuilder
from dugout.production.features.definitions import FEATURE_COLUMNS
from dugout.production.config import MODEL_DIR
from dugout.production.models.squad import FreeHitOptimizer
from dugout.production.models.predict import predict_points, get_last_model_type

# Signals that are FORBIDDEN in free hit decision (research-rejected)
FORBIDDEN_SIGNALS = {"p_play", "p60", "weighted_ev", "availability_weight"}

# Fixture-based signals that are FORBIDDEN (research-rejected)
FORBIDDEN_FIXTURE_SIGNALS = {"fixture_difficulty", "fdr", "opponent_strength", "fixture_weight", "fixture_adjustment"}


def _validate_no_research_imports():
    """Runtime assertion: no research module imports allowed.
    
    NOTE: Skipped during pytest runs to allow mixed test suites.
    """
    import sys
    
    # Skip check during pytest (research tests may have run first)
    if "pytest" in sys.modules:
        return
    
    research_modules = [m for m in sys.modules if "dugout.research" in m]
    if research_modules:
        raise RuntimeError(
            f"Contract violation: research modules imported in production: {research_modules}. "
            "Production decisions must not depend on research code."
        )


def optimize_free_hit(
    gw: Optional[int] = None,
    budget: float = 100.0,
    reader: Optional[DataReader] = None,
) -> Tuple[Any, pd.DataFrame, int, str]:
    """Optimize a Free Hit squad for a gameweek.
    
    Args:
        gw: Target gameweek (default: next GW)
        budget: Budget in millions (default: 100)
        reader: Optional DataReader instance (for testing)
    
    Returns:
        Tuple of (result, predictions_df, target_gw, model_type) where:
        - result: FreeHitResult with starting_xi, bench, total_ev, etc.
        - predictions_df: Full DataFrame with predictions
        - target_gw: The target gameweek
        - model_type: "two_stage" or "legacy"
    """
    # Runtime contract validation
    _validate_no_research_imports()
    
    if reader is None:
        reader = DataReader()
    
    raw_df = reader.get_all_gw_data()
    available_gws = sorted(raw_df["gw"].unique())
    latest_gw = available_gws[-1]
    
    # Determine target GW
    if gw is not None:
        target_gw = gw
        history_gw = gw - 1
    else:
        target_gw = latest_gw + 1
        history_gw = latest_gw
    
    if history_gw not in available_gws:
        raise ValueError(f"GW{history_gw} data not available. Cannot optimize for GW{target_gw}.")
    
    # Filter to data up to history_gw
    raw_df = raw_df[raw_df["gw"] <= history_gw].copy()
    
    # Build features
    fb = FeatureBuilder()
    features_df = fb.build_training_set(raw_df)
    latest_df = features_df[features_df["gw"] == history_gw].copy()
    
    # Runtime assertion: no fixture signals in features used for prediction
    fixture_present = FORBIDDEN_FIXTURE_SIGNALS.intersection(latest_df.columns)
    if fixture_present:
        raise RuntimeError(
            f"Contract violation: fixture signals detected in features: {fixture_present}. "
            "Free hit optimization must not use fixture-based weighting in predictions."
        )
    
    # Merge status
    meta_df = raw_df[raw_df["gw"] == history_gw][
        ["player_id", "status"]
    ].drop_duplicates()
    latest_df = latest_df.merge(meta_df, on="player_id", how="left")
    
    # Get FUTURE fixtures (target GW) - for DISPLAY only, not prediction
    fixtures = reader.get_fixtures(gw=target_gw)
    teams = reader.get_teams()
    team_short = {t["id"]: t["short_name"] for t in teams}
    
    # Build team_id -> opponent mapping (display only)
    fixture_map = {}
    for f in fixtures:
        h, a = f["team_h"], f["team_a"]
        fixture_map[h] = {"opponent_short": team_short.get(a, "?"), "is_home": True}
        fixture_map[a] = {"opponent_short": team_short.get(h, "?"), "is_home": False}
    
    latest_df["opponent_short"] = latest_df["team_id"].map(
        lambda x: fixture_map.get(x, {}).get("opponent_short", "?")
    )
    latest_df["is_home"] = latest_df["team_id"].map(
        lambda x: fixture_map.get(x, {}).get("is_home", False)
    )
    
    # Filter unavailable
    unavailable = ["n", "i", "s", "u"]
    latest_df = latest_df[~latest_df["status"].isin(unavailable)].copy()
    
    # Predict using unified interface (supports two-stage and legacy)
    # GUARDRAIL: This is the ONLY place predictions are made
    latest_df["predicted_points"] = predict_points(latest_df)
    model_type = get_last_model_type()
    
    # Contract assertion - verify no forbidden signals leaked in
    forbidden_present = FORBIDDEN_SIGNALS.intersection(latest_df.columns)
    if forbidden_present:
        raise RuntimeError(
            f"Contract violation: forbidden signals present: {forbidden_present}. "
            "Free hit must use LP maximize Σ(predicted_points) only."
        )
    
    # Prepare columns for optimizer
    latest_df["cost"] = latest_df["now_cost"]
    latest_df["name"] = latest_df["player_name"]
    latest_df["team"] = latest_df["team_name"]
    latest_df["element_type"] = latest_df["position"]
    
    # Add model_type to output for logging
    latest_df["model_type"] = model_type
    
    # Run optimizer with FROZEN settings: pure EV maximization
    optimizer = FreeHitOptimizer(
        predictions_df=latest_df,
        budget=budget,
    )
    result = optimizer.optimize()
    
    return result, latest_df, target_gw, model_type
