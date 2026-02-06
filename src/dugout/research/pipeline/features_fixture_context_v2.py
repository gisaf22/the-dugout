"""
Stage 4c — Fixture / Opponent Context (v2)

Constructs opponent-aware features using xG/xGC data from the gameweeks table.
Features are role-aware: attacking context for MID/FWD, defensive context for DEF/GKP.

Output: storage/research/datasets/features_fixture_context_v2.csv

Features (8 total):
    Attacking context (MID/FWD only):
        opp_xgc_per90_5    — Opponent's xGC per 90 (rolling 5 GWs)
        opp_cs_rate_5      — Opponent's clean sheet rate (rolling 5 GWs)
    
    Defensive context (DEF/GKP only):
        opp_xg_per90_5     — Opponent's xG per 90 (rolling 5 GWs)
    
    Game environment (all positions):
        match_total_xg     — team_xg_per90_5 + opp_xg_per90_5 (proxy for open game)
    
    Home/Away (all positions):
        is_home            — 1 if playing at home, 0 if away

Construction Rules:
    - Features use ONLY data from GWs ≤ t-1 (temporal contract enforced via shift)
    - Rolling window of 5 matches, min_periods=1 for early GWs
    - One row per (player_id, gw)
    - Role-specific features are NaN for non-applicable positions
    - Joined to targets.csv to preserve target alignment

Guarantees:
    - Deterministic
    - Idempotent
    - No temporal leakage
"""

import sqlite3
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

WINDOW_SIZE = 5

# Position codes: 1=GKP, 2=DEF, 3=MID, 4=FWD
ATTACKING_POSITIONS = [3, 4]  # MID, FWD
DEFENSIVE_POSITIONS = [1, 2]  # GKP, DEF

# Feature columns by category
ATTACKING_FEATURES = ["opp_xgc_per90_5", "opp_cs_rate_5"]
DEFENSIVE_FEATURES = ["opp_xg_per90_5"]
ENVIRONMENT_FEATURES = ["match_total_xg"]
HOME_FEATURES = ["is_home"]

FEATURE_COLUMNS = ATTACKING_FEATURES + DEFENSIVE_FEATURES + ENVIRONMENT_FEATURES + HOME_FEATURES
OUTPUT_COLUMNS = ["player_id", "gw"] + FEATURE_COLUMNS


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def load_team_gw_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load team-level xG and xGC per gameweek.
    
    Uses MAX(xGC) and SUM(xG) to get team totals from player-level data.
    MAX is used for xGC because it's duplicated across players (scaled by minutes).
    SUM is used for xG because each player has their own xG.
    
    Returns: team_id, gw, team_xg, team_xgc, team_cs
    """
    query = """
    SELECT 
        team_id,
        round AS gw,
        SUM(expected_goals) AS team_xg,
        MAX(expected_goals_conceded) AS team_xgc,
        MAX(clean_sheets) AS team_cs
    FROM gameweeks
    WHERE minutes > 0
    GROUP BY team_id, round
    ORDER BY team_id, round
    """
    return pd.read_sql(query, conn)


def load_fixtures(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load finished fixtures to determine home/away and opponent.
    
    Returns: gw, team_h, team_a
    """
    query = """
    SELECT 
        event AS gw,
        team_h,
        team_a
    FROM fixtures
    WHERE finished = 1
    ORDER BY event
    """
    return pd.read_sql(query, conn)


def load_player_team_position_mapping(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load player-team-position-gw mapping from gameweeks table.
    
    Returns: player_id, gw, team_id, element_type (position)
    """
    query = """
    SELECT DISTINCT
        element_id AS player_id,
        round AS gw,
        team_id,
        element_type
    FROM gameweeks
    ORDER BY element_id, round
    """
    return pd.read_sql(query, conn)


def load_targets(targets_path: Path) -> pd.DataFrame:
    """Load the target table to join features onto."""
    return pd.read_csv(targets_path)


# -----------------------------------------------------------------------------
# Rolling Team Stats Computation
# -----------------------------------------------------------------------------


def compute_rolling_team_stats(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling averages for team xG, xGC, and CS rate.
    
    CRITICAL: Shift by 1 to ensure feature at GW t uses only data from GWs ≤ t-1.
    
    Returns: team_id, gw, xg_per90_5, xgc_per90_5, cs_rate_5
    """
    results = []
    
    for team_id, team_df in team_stats.groupby("team_id"):
        team_df = team_df.sort_values("gw").reset_index(drop=True)
        
        # Rolling stats
        rolling_xg = team_df["team_xg"].rolling(WINDOW_SIZE, min_periods=1).mean()
        rolling_xgc = team_df["team_xgc"].rolling(WINDOW_SIZE, min_periods=1).mean()
        rolling_cs = team_df["team_cs"].rolling(WINDOW_SIZE, min_periods=1).mean()
        
        # SHIFT BY 1: Temporal contract enforcement
        # After shift, row at GW t has stats from GWs ending at t-1
        team_result = pd.DataFrame({
            "team_id": team_id,
            "gw": team_df["gw"],
            "xg_per90_5": rolling_xg.shift(1),
            "xgc_per90_5": rolling_xgc.shift(1),
            "cs_rate_5": rolling_cs.shift(1),
        })
        results.append(team_result)
    
    return pd.concat(results, ignore_index=True)


# -----------------------------------------------------------------------------
# Fixture Context Feature Construction
# -----------------------------------------------------------------------------


def compute_fixture_context_features_v2(
    player_team_pos: pd.DataFrame,
    fixtures: pd.DataFrame,
    rolling_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute fixture context features for all players.
    
    Steps:
    1. For each (player, gw), find opponent team and home/away status
    2. Look up opponent's rolling stats
    3. Look up player's team rolling stats (for match_total_xg)
    4. Apply role-specific masking
    
    Returns: player_id, gw, + all FEATURE_COLUMNS
    """
    # Step 1: Join player's team to fixtures to determine opponent and home/away
    player_fixtures = player_team_pos.merge(
        fixtures[["gw", "team_h", "team_a"]],
        on="gw",
        how="inner"
    )
    
    # Keep only fixtures where player's team is involved
    player_fixtures = player_fixtures[
        (player_fixtures["team_id"] == player_fixtures["team_h"]) |
        (player_fixtures["team_id"] == player_fixtures["team_a"])
    ].copy()
    
    # Determine opponent and is_home
    player_fixtures["opponent_id"] = player_fixtures.apply(
        lambda row: row["team_a"] if row["team_id"] == row["team_h"] else row["team_h"],
        axis=1
    )
    player_fixtures["is_home"] = (
        player_fixtures["team_id"] == player_fixtures["team_h"]
    ).astype(int)
    
    # Step 2: Look up opponent's rolling stats
    opp_stats = rolling_stats.rename(columns={
        "team_id": "opponent_id",
        "xg_per90_5": "opp_xg_per90_5",
        "xgc_per90_5": "opp_xgc_per90_5",
        "cs_rate_5": "opp_cs_rate_5",
    })
    
    features = player_fixtures.merge(
        opp_stats[["opponent_id", "gw", "opp_xg_per90_5", "opp_xgc_per90_5", "opp_cs_rate_5"]],
        on=["opponent_id", "gw"],
        how="left"
    )
    
    # Step 3: Look up player's team rolling stats (for match_total_xg)
    team_stats = rolling_stats[["team_id", "gw", "xg_per90_5"]].rename(
        columns={"xg_per90_5": "team_xg_per90_5"}
    )
    
    features = features.merge(
        team_stats,
        on=["team_id", "gw"],
        how="left"
    )
    
    # Compute match_total_xg (proxy for open game)
    features["match_total_xg"] = features["team_xg_per90_5"] + features["opp_xg_per90_5"]
    
    # Step 4: Apply role-specific masking
    # Attacking features (opp_xgc_per90_5, opp_cs_rate_5) only for MID/FWD
    # Defensive features (opp_xg_per90_5) only for DEF/GKP
    
    # Mask attacking features for non-attacking positions
    attacking_mask = ~features["element_type"].isin(ATTACKING_POSITIONS)
    features.loc[attacking_mask, ATTACKING_FEATURES] = None
    
    # Mask defensive features for non-defensive positions
    defensive_mask = ~features["element_type"].isin(DEFENSIVE_POSITIONS)
    features.loc[defensive_mask, DEFENSIVE_FEATURES] = None
    
    # Select output columns
    output = features[["player_id", "gw"] + FEATURE_COLUMNS].copy()
    
    # Deduplicate (shouldn't happen, but safety)
    output = output.drop_duplicates(subset=["player_id", "gw"])
    
    return output


# -----------------------------------------------------------------------------
# Contract Enforcement
# -----------------------------------------------------------------------------


def enforce_feature_contract(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    """
    Enforce Stage 4c contract on fixture context features v2.
    
    Raises AssertionError if any check fails.
    
    Checks:
        1. (player_id, gw) uniqueness preserved
        2. Numeric features are non-negative where not NaN
        3. First GW should have NaN for rolling features (no prior data)
        4. is_home is binary (0 or 1)
    """
    # Check 1: Uniqueness
    duplicates = features.duplicated(subset=["player_id", "gw"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (player_id, gw) pairs"
    
    # Check 2: Non-negative numeric features
    for col in ["opp_xgc_per90_5", "opp_cs_rate_5", "opp_xg_per90_5", "match_total_xg"]:
        valid = features[col].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all(), f"{col} has negative values"
    
    # Check 3: First GW should have NaN for rolling features
    gw1 = features[features["gw"] == 1]
    rolling_cols = ["opp_xgc_per90_5", "opp_cs_rate_5", "opp_xg_per90_5", "match_total_xg"]
    for col in rolling_cols:
        assert gw1[col].isna().all(), f"GW 1 should have all NaN for {col} (no prior data)"
    
    # Check 4: is_home is binary
    valid_home = features["is_home"].dropna()
    assert set(valid_home.unique()).issubset({0, 1}), "is_home must be 0 or 1"


# -----------------------------------------------------------------------------
# Pipeline Entry Point
# -----------------------------------------------------------------------------


def build_fixture_context_features_v2(
    db_path: Path,
    targets_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Stage 4c pipeline entry point.
    
    Loads data, computes fixture context features v2, joins to targets,
    validates, and saves.
    
    Args:
        db_path: Path to SQLite database
        targets_path: Path to targets.csv from Stage 2
        output_path: Path to write features_fixture_context_v2.csv
        
    Returns:
        The validated features DataFrame
    """
    # Load data
    conn = sqlite3.connect(db_path)
    try:
        team_stats = load_team_gw_stats(conn)
        fixtures = load_fixtures(conn)
        player_team_pos = load_player_team_position_mapping(conn)
    finally:
        conn.close()
    
    targets = load_targets(targets_path)
    
    # Compute rolling team stats
    rolling_stats = compute_rolling_team_stats(team_stats)
    
    # Compute features
    features = compute_fixture_context_features_v2(player_team_pos, fixtures, rolling_stats)
    
    # Join to targets (ensures alignment with target table)
    merged = targets[["player_id", "gw"]].merge(
        features,
        on=["player_id", "gw"],
        how="left",
    )
    
    # Ensure output columns exist
    for col in FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = None
    
    # Select output columns
    output = merged[OUTPUT_COLUMNS]
    
    # Enforce contract
    enforce_feature_contract(output, targets)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    
    return output


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for Stage 4c."""
    project_root = Path(__file__).resolve().parents[4]
    db_path = project_root / "storage" / "fpl_2025_26.sqlite"
    targets_path = project_root / "storage" / "research" / "datasets" / "targets.csv"
    output_path = project_root / "storage" / "research" / "datasets" / "features_fixture_context_v2.csv"
    
    print("=" * 60)
    print("Stage 4c — Fixture Context Feature Construction (v2)")
    print("=" * 60)
    print(f"  Database: {db_path}")
    print(f"  Targets:  {targets_path}")
    print(f"  Output:   {output_path}")
    
    features = build_fixture_context_features_v2(db_path, targets_path, output_path)
    
    # Summary stats
    print("\nCompleted.")
    print(f"  Rows:    {len(features):,}")
    print(f"  Players: {features['player_id'].nunique():,}")
    print(f"  GWs:     {features['gw'].min()} to {features['gw'].max()}")
    
    # Feature coverage (GW >= 2)
    gw2_plus = features[features["gw"] >= 2]
    print(f"\nFeature coverage (GW >= 2):")
    for col in FEATURE_COLUMNS:
        non_null = gw2_plus[col].notna().sum()
        print(f"  {col}: {non_null:,} / {len(gw2_plus):,} ({non_null/len(gw2_plus):.1%})")
    
    # Feature statistics
    print(f"\nFeature statistics:")
    for col in FEATURE_COLUMNS:
        valid = features[col].dropna()
        if len(valid) > 0:
            print(f"  {col}:")
            print(f"    Mean: {valid.mean():.3f}, Std: {valid.std():.3f}")
            print(f"    Min:  {valid.min():.3f}, Max: {valid.max():.3f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
