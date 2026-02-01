"""
Stage 4b — Fixture Context Feature Construction

Constructs exactly ONE fixture-level contextual feature for each (player_id, gw):
    opp_def_strength — opponent team's rolling average goals conceded per match

This feature measures how leaky the opponent's defense has been recently.
- Higher value → weaker defense (concedes more goals)
- Lower value → stronger defense (concedes fewer goals)

Output: storage/datasets/features_fixture_context.csv

Schema:
    player_id        int     Player identifier
    gw               int     Target gameweek (t)
    opp_def_strength float   Opponent's avg goals conceded (last 5 matches)

Construction Rules:
    - Feature uses ONLY data from GWs ≤ t-1 (temporal contract enforced via shift)
    - Rolling window of 5 matches, min_periods=1 for early GWs
    - One row per (player_id, gw)
    - Joined to targets.csv to preserve target alignment

Guarantees:
    - Deterministic
    - Idempotent
    - No temporal leakage
"""

import sqlite3
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

WINDOW_SIZE = 5
FEATURE_COLUMNS = ["opp_def_strength"]
OUTPUT_COLUMNS = ["player_id", "gw"] + FEATURE_COLUMNS


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def load_fixtures(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load finished fixtures with scores.
    
    Returns gw, team_h, team_a, team_h_score, team_a_score.
    """
    query = """
    SELECT 
        event AS gw,
        team_h,
        team_a,
        team_h_score,
        team_a_score
    FROM fixtures
    WHERE finished = 1
    ORDER BY event
    """
    return pd.read_sql(query, conn)


def load_player_team_mapping(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load player-team-gw mapping from gameweeks table.
    
    Returns player_id, gw, team_id.
    """
    query = """
    SELECT DISTINCT
        element_id AS player_id,
        round AS gw,
        team_id
    FROM gameweeks
    ORDER BY element_id, round
    """
    return pd.read_sql(query, conn)


def load_targets(targets_path: Path) -> pd.DataFrame:
    """Load the target table to join features onto."""
    return pd.read_csv(targets_path)


# -----------------------------------------------------------------------------
# Team Goals Conceded Computation
# -----------------------------------------------------------------------------


def compute_team_goals_conceded(fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Compute goals conceded per team per GW.
    
    Each fixture produces TWO rows:
    - Home team concedes team_a_score
    - Away team concedes team_h_score
    
    Returns: team_id, gw, goals_conceded
    """
    # Home team's goals conceded = away team's score
    home_conceded = fixtures[["gw", "team_h", "team_a_score"]].rename(columns={
        "team_h": "team_id",
        "team_a_score": "goals_conceded"
    })
    
    # Away team's goals conceded = home team's score
    away_conceded = fixtures[["gw", "team_a", "team_h_score"]].rename(columns={
        "team_a": "team_id",
        "team_h_score": "goals_conceded"
    })
    
    # Combine and sort
    conceded = pd.concat([home_conceded, away_conceded], ignore_index=True)
    conceded = conceded.sort_values(["team_id", "gw"]).reset_index(drop=True)
    
    return conceded


def compute_rolling_defensive_strength(conceded: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling average goals conceded per team.
    
    CRITICAL: Shift by 1 to ensure feature at GW t uses only data from GWs ≤ t-1.
    
    Returns: team_id, gw, avg_goals_conceded (shifted)
    """
    results = []
    
    for team_id, team_df in conceded.groupby("team_id"):
        team_df = team_df.sort_values("gw").reset_index(drop=True)
        
        # Rolling mean of goals conceded
        rolling_avg = team_df["goals_conceded"].rolling(
            WINDOW_SIZE, 
            min_periods=1
        ).mean()
        
        # SHIFT BY 1: Temporal contract enforcement
        # After shift, row at GW t has stats from GWs ending at t-1
        shifted = rolling_avg.shift(1)
        
        team_result = pd.DataFrame({
            "team_id": team_id,
            "gw": team_df["gw"],
            "avg_goals_conceded": shifted
        })
        results.append(team_result)
    
    return pd.concat(results, ignore_index=True)


# -----------------------------------------------------------------------------
# Feature Construction
# -----------------------------------------------------------------------------


def compute_fixture_context_features(
    player_team: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute fixture context features for all players.
    
    Steps:
    1. Compute goals conceded per team per GW
    2. Compute rolling defensive strength per team
    3. For each (player, gw), find opponent team
    4. Look up opponent's defensive strength
    
    Returns: player_id, gw, opp_def_strength
    """
    # Step 1: Goals conceded per team per GW
    conceded = compute_team_goals_conceded(fixtures)
    
    # Step 2: Rolling defensive strength
    def_strength = compute_rolling_defensive_strength(conceded)
    
    # Step 3: For each (player, gw), find opponent team
    # Join player's team to fixtures to determine opponent
    player_fixtures = player_team.merge(
        fixtures[["gw", "team_h", "team_a"]],
        on="gw",
        how="inner"
    )
    
    # Keep only fixtures where player's team is involved
    player_fixtures = player_fixtures[
        (player_fixtures["team_id"] == player_fixtures["team_h"]) |
        (player_fixtures["team_id"] == player_fixtures["team_a"])
    ].copy()
    
    # Determine opponent: if player's team is home, opponent is away (and vice versa)
    player_fixtures["opponent_id"] = player_fixtures.apply(
        lambda row: row["team_a"] if row["team_id"] == row["team_h"] else row["team_h"],
        axis=1
    )
    
    # Step 4: Look up opponent's defensive strength
    # The defensive strength at GW t is based on opponent's performance up to GW t-1
    # We need to join on opponent_id and gw
    features = player_fixtures[["player_id", "gw", "opponent_id"]].merge(
        def_strength.rename(columns={
            "team_id": "opponent_id",
            "avg_goals_conceded": "opp_def_strength"
        }),
        on=["opponent_id", "gw"],
        how="left"
    )
    
    # Select output columns
    features = features[["player_id", "gw", "opp_def_strength"]]
    
    # Deduplicate (shouldn't happen, but safety)
    features = features.drop_duplicates(subset=["player_id", "gw"])
    
    return features


# -----------------------------------------------------------------------------
# Contract Enforcement
# -----------------------------------------------------------------------------


def enforce_feature_contract(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    """
    Enforce Stage 4b contract on fixture context features.
    
    Raises AssertionError if any check fails.
    
    Checks:
        1. (player_id, gw) uniqueness preserved
        2. opp_def_strength >= 0 where not NaN
        3. First GW should have NaN (no prior opponent data)
    """
    # Check 1: Uniqueness
    duplicates = features.duplicated(subset=["player_id", "gw"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (player_id, gw) pairs"
    
    # Check 2: Non-negative defensive strength
    valid = features["opp_def_strength"].dropna()
    assert (valid >= 0).all(), "opp_def_strength has negative values"
    
    # Check 3: First GW should have NaN (opponent hasn't played yet)
    gw1 = features[features["gw"] == 1]["opp_def_strength"]
    assert gw1.isna().all(), "GW 1 should have all NaN (no prior opponent data)"


# -----------------------------------------------------------------------------
# Pipeline Entry Point
# -----------------------------------------------------------------------------


def build_fixture_context_features(
    db_path: Path,
    targets_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Stage 4b pipeline entry point.
    
    Loads fixtures, computes fixture context features, joins to targets,
    validates, and saves.
    
    Args:
        db_path: Path to SQLite database
        targets_path: Path to targets.csv from Stage 2
        output_path: Path to write features_fixture_context.csv
        
    Returns:
        The validated features DataFrame
    """
    # Load data
    conn = sqlite3.connect(db_path)
    try:
        fixtures = load_fixtures(conn)
        player_team = load_player_team_mapping(conn)
    finally:
        conn.close()
    
    targets = load_targets(targets_path)
    
    # Compute features
    features = compute_fixture_context_features(player_team, fixtures)
    
    # Join to targets (ensures alignment with target table)
    merged = targets[["player_id", "gw"]].merge(
        features,
        on=["player_id", "gw"],
        how="left",
    )
    
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
    """Command-line entry point for Stage 4b."""
    project_root = Path(__file__).resolve().parents[4]
    db_path = project_root / "storage" / "fpl_2025_26.sqlite"
    targets_path = project_root / "storage" / "research" / "datasets" / "targets.csv"
    output_path = project_root / "storage" / "research" / "datasets" / "features_fixture_context.csv"
    
    print("=" * 60)
    print("Stage 4b — Fixture Context Feature Construction")
    print("=" * 60)
    print(f"  Database: {db_path}")
    print(f"  Targets:  {targets_path}")
    print(f"  Output:   {output_path}")
    
    features = build_fixture_context_features(db_path, targets_path, output_path)
    
    # Summary stats
    print("\nCompleted.")
    print(f"  Rows:    {len(features):,}")
    print(f"  Players: {features['player_id'].nunique():,}")
    print(f"  GWs:     {features['gw'].min()} to {features['gw'].max()}")
    
    # Feature coverage
    gw2_plus = features[features["gw"] >= 2]
    non_null = gw2_plus["opp_def_strength"].notna().sum()
    print(f"\nFeature coverage (GW >= 2):")
    print(f"  opp_def_strength: {non_null:,} / {len(gw2_plus):,} ({non_null/len(gw2_plus):.1%})")
    
    # Feature statistics
    valid = features["opp_def_strength"].dropna()
    print(f"\nFeature statistics:")
    print(f"  Mean:   {valid.mean():.3f}")
    print(f"  Std:    {valid.std():.3f}")
    print(f"  Min:    {valid.min():.3f}")
    print(f"  Max:    {valid.max():.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
