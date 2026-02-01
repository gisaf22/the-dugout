"""
Stage 4a — Conditional Performance Feature Construction

Constructs performance features for each (player_id, gw) that describe
what a player does WHEN they play, conditional on minutes > 0.

These features use only historical data from GWs ≤ t-1 where the player
actually appeared. They are belief inputs, not predictions.

Output: storage/datasets/features_performance.csv

Schema:
    player_id         int     Player identifier
    gw                int     Target gameweek (t)
    points_per_90_5   float   Rolling mean of points per 90 (last 5 appearances)
    xGI_per_90_5      float   Rolling mean of (xG + xA) per 90
    bonus_per_90_5    float   Rolling mean of bonus per 90
    ict_per_90_5      float   Rolling mean of ICT index per 90

Construction Rules:
    - All features use ONLY GWs ≤ t-1 (temporal contract enforced via shift)
    - Only rows with minutes > 0 contribute to rolling calculations
    - Per-90 normalization: (stat / minutes) * 90
    - Rolling window of 5 appearances, with min_periods=1
    - One row per (player_id, gw)
    - Joined to targets.csv to preserve target alignment

Guarantees:
    - Deterministic
    - Idempotent
    - No temporal leakage (no GW t data used)
    - Conditional on participation (minutes > 0 only)
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

WINDOW_SIZE = 5
FEATURE_COLUMNS = [
    "points_per_90_5",
    "xGI_per_90_5",
    "bonus_per_90_5",
    "ict_per_90_5",
]
OUTPUT_COLUMNS = ["player_id", "gw"] + FEATURE_COLUMNS


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def load_gameweek_outcomes(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load raw gameweek outcomes from the database.

    Returns columns needed for conditional performance feature computation.
    No filtering, no aggregation, no lagging.
    """
    query = """
    SELECT 
        element_id AS player_id,
        round AS gw,
        minutes,
        total_points,
        expected_goals,
        expected_assists,
        bonus,
        ict_index
    FROM gameweeks
    ORDER BY element_id, round
    """
    return pd.read_sql(query, conn)


def load_targets(targets_path: Path) -> pd.DataFrame:
    """Load the target table to join features onto."""
    return pd.read_csv(targets_path)


# -----------------------------------------------------------------------------
# Per-90 Computation
# -----------------------------------------------------------------------------


def compute_per_90(stat: pd.Series, minutes: pd.Series) -> pd.Series:
    """
    Compute per-90 normalized stat.

    Args:
        stat: Raw stat values (points, xG, etc.)
        minutes: Minutes played

    Returns:
        Per-90 normalized values (NaN where minutes == 0)
    """
    # Avoid division by zero; per-90 is undefined for 0 minutes
    return np.where(minutes > 0, (stat / minutes) * 90, np.nan)


# -----------------------------------------------------------------------------
# Feature Computation
# -----------------------------------------------------------------------------


def compute_player_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute conditional performance features for a single player.

    CRITICAL:
    1. Only appearances (minutes > 0) contribute to rolling stats
    2. Rolling window is over APPEARANCES, not GWs
    3. All rolling stats are shifted by 1 to ensure features at GW t
       use only data from GWs ≤ t-1

    Args:
        player_df: DataFrame for one player, sorted by gw

    Returns:
        DataFrame with player_id, gw, and feature columns for ALL GWs
        (features are NaN for GWs before first appearance)
    """
    df = player_df.copy()

    # Compute per-90 stats (NaN for minutes == 0)
    df["points_per_90"] = compute_per_90(df["total_points"], df["minutes"])
    df["xGI_per_90"] = compute_per_90(
        df["expected_goals"] + df["expected_assists"], df["minutes"]
    )
    df["bonus_per_90"] = compute_per_90(df["bonus"], df["minutes"])
    df["ict_per_90"] = compute_per_90(df["ict_index"], df["minutes"])

    # Mask for appearances only
    played_mask = df["minutes"] > 0

    per_90_cols = ["points_per_90", "xGI_per_90", "bonus_per_90", "ict_per_90"]

    for col in per_90_cols:
        # Compute rolling mean over APPEARANCES only (not GWs)
        # This ensures window = last 5 appearances, not last 5 GWs
        rolling_mean = (
            df.loc[played_mask, col]
            .rolling(WINDOW_SIZE, min_periods=1)
            .mean()
        )

        # Align back to full GW index (non-appearances get NaN)
        rolling_mean = rolling_mean.reindex(df.index)

        # Forward-fill to propagate last known value to non-playing GWs
        rolling_mean = rolling_mean.ffill()

        # Shift by 1: feature at GW t uses data through GW t-1
        df[f"{col}_shifted"] = rolling_mean.shift(1)

    # Build output DataFrame
    return pd.DataFrame({
        "player_id": df["player_id"].values,
        "gw": df["gw"].values,
        "points_per_90_5": df["points_per_90_shifted"].values,
        "xGI_per_90_5": df["xGI_per_90_shifted"].values,
        "bonus_per_90_5": df["bonus_per_90_shifted"].values,
        "ict_per_90_5": df["ict_per_90_shifted"].values,
    })


def compute_performance_features(gw_outcomes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute conditional performance features for all players.

    Args:
        gw_outcomes: DataFrame with player_id, gw, minutes, and stat columns

    Returns:
        DataFrame with player_id, gw, and all feature columns
    """
    # Sort to ensure proper rolling order within each player
    df = gw_outcomes.sort_values(["player_id", "gw"]).reset_index(drop=True)

    # Compute features per player
    features_list = []
    for _, player_df in df.groupby("player_id"):
        player_features = compute_player_features(player_df)
        features_list.append(player_features)

    features = pd.concat(features_list, ignore_index=True)
    return features


# -----------------------------------------------------------------------------
# Contract Enforcement
# -----------------------------------------------------------------------------


def enforce_feature_contract(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    """
    Enforce Stage 4a contract on performance features.

    Raises AssertionError if any check fails.

    Checks:
        1. (player_id, gw) uniqueness preserved from targets
        2. Per-90 features are finite where not NaN
        3. Non-negative constraint only for xGI, bonus, ICT (points can be negative)
        4. Features are NaN only for player's first GW or when no prior appearances
        5. No feature uses GW t data (enforced by shift, validated by structure)
    """
    # Check 1: Uniqueness preserved
    duplicates = features.duplicated(subset=["player_id", "gw"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (player_id, gw) pairs"

    # Check 2: Per-90 features are finite where not NaN
    for col in FEATURE_COLUMNS:
        valid = features[col].dropna()
        assert np.isfinite(valid).all(), f"{col} has non-finite values"

    # Check 3: Non-negative constraint for features that can't be negative
    # Note: points_per_90 CAN be negative (own goals, red cards, etc.)
    non_negative_cols = ["xGI_per_90_5", "bonus_per_90_5", "ict_per_90_5"]
    for col in non_negative_cols:
        valid = features[col].dropna()
        assert (valid >= 0).all(), f"{col} has negative values"

    # Check 4: Player's first GW should have NaN features (no prior data)
    first_gw = features.groupby("player_id")["gw"].transform("min")
    first_gw_rows = features[features["gw"] == first_gw]
    for col in FEATURE_COLUMNS:
        first_gw_non_null = first_gw_rows[col].notna().sum()
        assert first_gw_non_null == 0, (
            f"First GW rows should have NaN for {col}, "
            f"but found {first_gw_non_null} non-NaN"
        )

    # Check 5: Row count matches targets
    assert len(features) == len(targets), (
        f"Feature row count ({len(features)}) != target row count ({len(targets)})"
    )


# -----------------------------------------------------------------------------
# Pipeline Entry Point
# -----------------------------------------------------------------------------


def build_performance_features(
    db_path: Path,
    targets_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Stage 4a pipeline entry point.

    Loads gameweek data, computes conditional performance features,
    joins to targets, validates, and saves.

    Args:
        db_path: Path to SQLite database
        targets_path: Path to targets.csv from Stage 2
        output_path: Path to write features_performance.csv

    Returns:
        The validated features DataFrame
    """
    # Load data
    conn = sqlite3.connect(db_path)
    try:
        gw_outcomes = load_gameweek_outcomes(conn)
    finally:
        conn.close()

    targets = load_targets(targets_path)

    # Compute features
    features = compute_performance_features(gw_outcomes)

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
    """Command-line entry point for Stage 4a."""
    # Resolve paths: features_performance.py -> pipeline -> dugout -> src -> project_root
    project_root = Path(__file__).resolve().parents[4]
    db_path = project_root / "storage" / "fpl_2025_26.sqlite"
    targets_path = project_root / "storage" / "research" / "datasets" / "targets.csv"
    output_path = project_root / "storage" / "research" / "datasets" / "features_performance.csv"

    print("Stage 4a — Conditional Performance Feature Construction")
    print(f"  Database: {db_path}")
    print(f"  Targets:  {targets_path}")
    print(f"  Output:   {output_path}")

    features = build_performance_features(db_path, targets_path, output_path)

    # Summary stats
    print("\nCompleted.")
    print(f"  Rows:    {len(features):,}")
    print(f"  Players: {features['player_id'].nunique():,}")
    print(f"  GWs:     {features['gw'].min()} to {features['gw'].max()}")

    # Feature coverage (exclude first GW per player)
    first_gw = features.groupby("player_id")["gw"].transform("min")
    has_prior = features["gw"] > first_gw
    valid_rows = features[has_prior]

    print(f"\nFeature coverage (rows with prior data: {len(valid_rows):,}):")
    for col in FEATURE_COLUMNS:
        non_null = valid_rows[col].notna().sum()
        print(f"  {col}: {non_null:,} / {len(valid_rows):,} ({non_null/len(valid_rows):.1%})")


if __name__ == "__main__":
    main()
