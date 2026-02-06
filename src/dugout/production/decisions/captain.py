"""Captain decision function.

Decision Rule (Frozen): argmax(predicted_points)
Validated by research pipeline Stage 6d.

Predictor meaning: μ(points | plays), position-conditional
Uses CaptainModel with defensive features for DEF/GKP.

CONTRACT ENFORCEMENT:
    - Any modification requires updating DECISION_CONTRACT_LAYER.md
    - Any modification requires re-running research pipeline Stage 6
    
GUARDRAIL: Must use CaptainModel from dugout.production.models.captain_model.
           No shared prediction code paths.
"""

import pandas as pd
from typing import Tuple, Optional

from dugout.production.data.reader import DataReader
from dugout.production.features.builder import FeatureBuilder
from dugout.production.config import MODEL_DIR
from dugout.production.models.registry import get_model

# Signals that are FORBIDDEN in captain decision (research-rejected)
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


def get_captain_candidates(
    gw: Optional[int] = None,
    top_n: int = 10,
    reader: Optional[DataReader] = None,
) -> Tuple[pd.DataFrame, int, str]:
    """Get captain candidates for a gameweek.
    
    Args:
        gw: Target gameweek (default: next GW)
        top_n: Number of candidates to return
        reader: Optional DataReader instance (for testing)
    
    Returns:
        Tuple of (candidates_df, target_gw, model_type) where:
        - candidates_df has columns: player_id, player_name, team_name, predicted_points, position
        - target_gw: the gameweek being predicted
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
        raise ValueError(f"GW{history_gw} data not available. Cannot predict GW{target_gw}.")
    
    # Filter to data up to history_gw
    raw_df = raw_df[raw_df["gw"] <= history_gw].copy()
    
    # Get FUTURE fixtures (target GW) for is_home_next feature
    fixtures = reader.get_fixtures(gw=target_gw)
    teams = reader.get_teams()
    team_short = {t["id"]: t["short_name"] for t in teams}
    
    # Build fixture maps
    fixture_map = {}      # team_id -> is_home (for FeatureBuilder)
    display_map = {}      # team_id -> {opponent_short, is_home} (for display)
    for f in fixtures:
        h, a = f["team_h"], f["team_a"]
        fixture_map[h] = True
        fixture_map[a] = False
        display_map[h] = {"opponent_short": team_short.get(a, "?"), "is_home": True}
        display_map[a] = {"opponent_short": team_short.get(h, "?"), "is_home": False}
    
    # Build features with correct is_home_next from target GW fixtures
    fb = FeatureBuilder()
    latest_df = fb.build_for_prediction(raw_df, fixture_map)
    
    # Runtime assertion: no fixture signals in features
    fixture_present = FORBIDDEN_FIXTURE_SIGNALS.intersection(latest_df.columns)
    if fixture_present:
        raise RuntimeError(
            f"Contract violation: fixture signals detected in features: {fixture_present}. "
            "Captain decision must not use fixture-based weighting."
        )
    
    # Merge status
    player_status = raw_df[raw_df["gw"] == history_gw][
        ["player_id", "status"]
    ].drop_duplicates()
    latest_df = latest_df.merge(player_status, on="player_id", how="left")
    
    # Filter unavailable
    unavailable = ["n", "i", "s", "u"]
    latest_df = latest_df[~latest_df["status"].isin(unavailable)].copy()
    
    # Load captain-specific model (position-conditional defensive features)
    # GUARDRAIL: This is the ONLY place captain predictions are made
    # See docs/research/decision_specific_modeling.md for ablation evidence
    try:
        captain_model = get_model("captain")
        latest_df["predicted_points"] = captain_model.predict(latest_df)
        model_type = "captain"
    except FileNotFoundError:
        # Fallback: use legacy predict_points if captain model not trained
        from dugout.production.models.predict import predict_points, get_last_model_type
        latest_df["predicted_points"] = predict_points(latest_df, model_variant="base")
        model_type = f"legacy_fallback ({get_last_model_type()})"
    
    # Add display columns (opponent, is_home for output only)
    latest_df["opponent_short"] = latest_df["team_id"].map(
        lambda x: display_map.get(x, {}).get("opponent_short", "?")
    )
    latest_df["is_home"] = latest_df["team_id"].map(
        lambda x: display_map.get(x, {}).get("is_home", False)
    )
    
    # Get top candidates
    candidates = latest_df.nlargest(top_n, "predicted_points")[
        ["player_id", "player_name", "team_name", "position", "predicted_points", "opponent_short", "is_home"]
    ].copy()
    
    # Add model_type to output for logging
    candidates["model_type"] = model_type
    
    return candidates, target_gw, model_type


def pick_captain(df: pd.DataFrame) -> pd.Series:
    """Apply frozen decision rule: argmax(predicted_points).
    
    CONTRACT ENFORCEMENT: This is the ONLY place captain decisions are made.
    Any modification requires re-validation via research pipeline.
    
    Args:
        df: DataFrame with 'predicted_points' column
        
    Returns:
        Series for the captain (row with highest predicted_points)
        
    Raises:
        RuntimeError: If contract is violated (missing/forbidden columns)
    """
    # Runtime contract validation
    _validate_no_research_imports()
    
    # Contract assertions - fail fast on violations
    if "predicted_points" not in df.columns:
        raise RuntimeError("Contract violation: 'predicted_points' column required")
    
    forbidden_present = FORBIDDEN_SIGNALS.intersection(df.columns)
    if forbidden_present:
        raise RuntimeError(
            f"Contract violation: forbidden signals present: {forbidden_present}. "
            f"Captain decision must use argmax(predicted_points) only."
        )
    
    fixture_present = FORBIDDEN_FIXTURE_SIGNALS.intersection(df.columns)
    if fixture_present:
        raise RuntimeError(
            f"Contract violation: fixture signals detected: {fixture_present}. "
            "Captain decision must not use fixture-based weighting."
        )
    
    # Frozen decision rule: argmax(predicted_points)
    return df.loc[df["predicted_points"].idxmax()]
