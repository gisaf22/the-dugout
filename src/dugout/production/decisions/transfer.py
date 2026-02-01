"""Transfer-in decision function.

Decision Rule (Frozen): argmax(predicted_points)
Validated by research pipeline Stage 7a.

CONTRACT ENFORCEMENT:
    - Any modification requires updating DECISION_CONTRACT_LAYER.md
    - Any modification requires re-running research pipeline Stage 7
    
GUARDRAIL: Must use predict_points() from dugout.production.models.predict.
           Direct model loading is FORBIDDEN.
"""

import pandas as pd
from typing import Tuple, Optional, Set

from dugout.production.data.reader import DataReader
from dugout.production.features.builder import FeatureBuilder
from dugout.production.features.definitions import FEATURE_COLUMNS
from dugout.production.config import MODEL_DIR
from dugout.production.models.predict import predict_points, get_last_model_type

# Signals that are FORBIDDEN in transfer decision (research-rejected)
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


def get_transfer_recommendations(
    gw: Optional[int] = None,
    top_n: int = 10,
    exclude_ids: Optional[Set[int]] = None,
    reader: Optional[DataReader] = None,
) -> Tuple[pd.DataFrame, int, str]:
    """Get transfer-in recommendations for a gameweek.
    
    Args:
        gw: Target gameweek (default: next GW)
        top_n: Number of recommendations to return
        exclude_ids: Player IDs to exclude (e.g., already owned)
        reader: Optional DataReader instance (for testing)
    
    Returns:
        Tuple of (recommendations_df, target_gw, model_type) where:
        - recommendations_df has columns: player_id, player_name, team_name, position, predicted_points, now_cost
        - target_gw: the gameweek being predicted
        - model_type: "two_stage" or "legacy"
    """
    # Runtime contract validation
    _validate_no_research_imports()
    
    if reader is None:
        reader = DataReader()
    
    if exclude_ids is None:
        exclude_ids = set()
    
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
        raise ValueError(f"GW{history_gw} data not available. Cannot predict GW{target_gw}.")
    
    # Filter to data up to history_gw
    raw_df = raw_df[raw_df["gw"] <= history_gw].copy()
    
    # Build features
    fb = FeatureBuilder()
    features_df = fb.build_training_set(raw_df)
    latest_df = features_df[features_df["gw"] == history_gw].copy()
    
    # Runtime assertion: no fixture signals in features
    fixture_present = FORBIDDEN_FIXTURE_SIGNALS.intersection(latest_df.columns)
    if fixture_present:
        raise RuntimeError(
            f"Contract violation: fixture signals detected in features: {fixture_present}. "
            "Transfer decision must not use fixture-based weighting."
        )
    
    # Merge status
    player_status = raw_df[raw_df["gw"] == history_gw][
        ["player_id", "status"]
    ].drop_duplicates()
    latest_df = latest_df.merge(player_status, on="player_id", how="left")
    
    # Filter unavailable
    unavailable = ["n", "i", "s", "u"]
    latest_df = latest_df[~latest_df["status"].isin(unavailable)].copy()
    
    # Exclude owned players
    if exclude_ids:
        latest_df = latest_df[~latest_df["player_id"].isin(exclude_ids)].copy()
    
    # Predict using unified interface (supports two-stage and legacy)
    # GUARDRAIL: This is the ONLY place predictions are made
    latest_df["predicted_points"] = predict_points(latest_df)
    model_type = get_last_model_type()
    
    # Contract assertion - verify no forbidden signals leaked in
    forbidden_present = FORBIDDEN_SIGNALS.intersection(latest_df.columns)
    if forbidden_present:
        raise RuntimeError(
            f"Contract violation: forbidden signals present: {forbidden_present}. "
            "Transfer decision must use argmax(predicted_points) only."
        )
    
    # Add fixture display info (DISPLAY ONLY - not used in ranking)
    latest_df = reader.enrich_with_fixture_display(latest_df, gw=target_gw, team_col="team_id")
    
    # Frozen decision rule: argmax(predicted_points)
    # Get top recommendations
    recommendations = latest_df.nlargest(top_n, "predicted_points")[
        ["player_id", "player_name", "team_name", "position", "predicted_points", "now_cost", "opponent_short", "is_home"]
    ].copy()
    
    # Add model_type to output for logging
    recommendations["model_type"] = model_type
    
    return recommendations, target_gw, model_type
