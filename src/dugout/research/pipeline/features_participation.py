"""
Stage 3 — Participation Feature Construction

Constructs empirical participation features for each (player_id, gw)
using only historical data from GWs ≤ t-1.

These features estimate a player's likelihood of appearing and reaching
60+ minutes. They are belief inputs, not decisions.

Output: storage/datasets/features_participation.csv

Schema:
    player_id              int     Player identifier
    gw                     int     Target gameweek (t)
    p_play_hat             float   P(minutes > 0) estimate from last 5 GWs
    p60_hat                float   P(minutes >= 60) estimate from last 5 GWs
    mins_std_5             float   Std of minutes over last 5 GWs
    mins_below_60_rate_5   float   Rate of minutes < 60 in last 5 GWs

Construction Rules:
    - All features use ONLY GWs ≤ t-1 (temporal contract enforced via shift)
    - Rolling window of 5, with min_periods=1 for early GWs
    - Window size is the actual count available, not fixed at 5
    - One row per (player_id, gw)
    - Joined to targets.csv to preserve target alignment

Guarantees:
    - Deterministic
    - Idempotent
    - No temporal leakage (no GW t data used)
"""

import sqlite3
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

WINDOW_SIZE = 5
FEATURE_COLUMNS = ["p_play_hat", "p60_hat", "mins_std_5", "mins_below_60_rate_5"]
OUTPUT_COLUMNS = ["player_id", "gw"] + FEATURE_COLUMNS


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def load_gameweek_outcomes(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load raw gameweek outcomes from the database.

    Returns player_id, gw, minutes for participation feature computation.
    No filtering, no aggregation, no lagging.
    """
    query = """
    SELECT 
        element_id AS player_id,
        round AS gw,
        minutes
    FROM gameweeks
    ORDER BY element_id, round
    """
    return pd.read_sql(query, conn)


def load_targets(targets_path: Path) -> pd.DataFrame:
    """Load the target table to join features onto."""
    return pd.read_csv(targets_path)


# -----------------------------------------------------------------------------
# Feature Computation
# -----------------------------------------------------------------------------


def compute_player_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute participation features for a single player.

    CRITICAL: All rolling stats are shifted by 1 to ensure features
    at GW t use only data from GWs ≤ t-1.

    Args:
        player_df: DataFrame for one player, sorted by gw

    Returns:
        DataFrame with player_id, gw, and feature columns
    """
    # Binary indicators on current GW outcomes
    played = (player_df["minutes"] > 0).astype(int)
    played_60 = (player_df["minutes"] >= 60).astype(int)
    below_60 = (player_df["minutes"] < 60).astype(int)
    minutes = player_df["minutes"]

    # Rolling computations (include current row, then shift to exclude it)
    # min_periods=1 allows computation with fewer than WINDOW_SIZE games
    roll_played = played.rolling(WINDOW_SIZE, min_periods=1).sum()
    roll_played_60 = played_60.rolling(WINDOW_SIZE, min_periods=1).sum()
    roll_below_60 = below_60.rolling(WINDOW_SIZE, min_periods=1).sum()
    # Count of GWs in window (not binary indicator values)
    roll_count = player_df["gw"].rolling(WINDOW_SIZE, min_periods=1).count()
    roll_mins_std = minutes.rolling(WINDOW_SIZE, min_periods=1).std()

    # SHIFT BY 1: This is the temporal contract enforcement
    # After shift, row at GW t has stats from GWs ending at t-1
    return pd.DataFrame({
        "player_id": player_df["player_id"].values,
        "gw": player_df["gw"].values,
        "p_play_hat": (roll_played / roll_count).shift(1).values,
        "p60_hat": (roll_played_60 / roll_count).shift(1).values,
        "mins_std_5": roll_mins_std.shift(1).values,
        "mins_below_60_rate_5": (roll_below_60 / roll_count).shift(1).values,
    })


def compute_participation_features(gw_outcomes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute participation features for all players.

    Args:
        gw_outcomes: DataFrame with player_id, gw, minutes

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
    Enforce Stage 3 contract on participation features.

    Raises AssertionError if any check fails.

    Checks:
        1. (player_id, gw) uniqueness preserved from targets
        2. Feature bounds: 0 <= p_play_hat <= 1, 0 <= p60_hat <= 1
        3. Logical: p60_hat <= p_play_hat
        4. No missing values except player's first GW (no prior data)
        5. mins_std_5 NaN allowed for first 2 GWs per player (need 2+ points for std)
    """
    # Check 1: Uniqueness preserved
    duplicates = features.duplicated(subset=["player_id", "gw"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (player_id, gw) pairs"

    # Identify each player's first GW (no prior data available)
    first_gw = features.groupby("player_id")["gw"].transform("min")
    has_prior_data = features["gw"] > first_gw

    # Filter to rows that should have valid rate features
    valid = features[has_prior_data].copy()

    # Check 2: Rate features in [0, 1]
    for col in ["p_play_hat", "p60_hat", "mins_below_60_rate_5"]:
        col_valid = valid[col].dropna()
        assert (col_valid >= 0).all(), f"{col} has values < 0"
        assert (col_valid <= 1).all(), f"{col} has values > 1"

    # Check 3: p60_hat <= p_play_hat (can't play 60+ without playing)
    valid_both = valid.dropna(subset=["p_play_hat", "p60_hat"])
    violations = (valid_both["p60_hat"] > valid_both["p_play_hat"] + 1e-9).sum()
    assert violations == 0, f"Found {violations} rows where p60_hat > p_play_hat"

    # Check 4: No missing rate features for rows with prior data
    for col in ["p_play_hat", "p60_hat", "mins_below_60_rate_5"]:
        missing = valid[col].isna().sum()
        assert missing == 0, f"{col} has {missing} missing values for rows with prior data"

    # Check 5: mins_std_5 needs 2+ prior GWs (second GW per player may still be NaN)
    second_gw = first_gw + 1
    has_2_prior = features["gw"] > second_gw
    valid_std = features[has_2_prior]["mins_std_5"]
    missing_std = valid_std.isna().sum()
    assert missing_std == 0, f"mins_std_5 has {missing_std} missing values for rows with 2+ prior GWs"

    # Check 6: Player's first GW should have all NaN features (no prior data)
    first_gw_rows = features[~has_prior_data]
    for col in FEATURE_COLUMNS:
        first_missing = first_gw_rows[col].isna().sum()
        assert first_missing == len(first_gw_rows), (
            f"First GW rows should have all NaN for {col}, "
            f"but found {len(first_gw_rows) - first_missing} non-NaN"
        )


# -----------------------------------------------------------------------------
# Pipeline Entry Point
# -----------------------------------------------------------------------------


def build_participation_features(
    db_path: Path,
    targets_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Stage 3 pipeline entry point.

    Loads gameweek data, computes participation features, joins to targets,
    validates, and saves.

    Args:
        db_path: Path to SQLite database
        targets_path: Path to targets.csv from Stage 2
        output_path: Path to write features_participation.csv

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
    features = compute_participation_features(gw_outcomes)

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
    """Command-line entry point for Stage 3."""
    # Resolve paths: features_participation.py -> pipeline -> dugout -> src -> project_root
    project_root = Path(__file__).resolve().parents[4]
    db_path = project_root / "storage" / "fpl_2025_26.sqlite"
    targets_path = project_root / "storage" / "research" / "datasets" / "targets.csv"
    output_path = project_root / "storage" / "research" / "datasets" / "features_participation.csv"

    print("Stage 3 — Participation Feature Construction")
    print(f"  Database: {db_path}")
    print(f"  Targets:  {targets_path}")
    print(f"  Output:   {output_path}")

    features = build_participation_features(db_path, targets_path, output_path)

    # Summary stats
    print("\nCompleted.")
    print(f"  Rows:    {len(features):,}")
    print(f"  Players: {features['player_id'].nunique():,}")
    print(f"  GWs:     {features['gw'].min()} to {features['gw'].max()}")

    # Feature coverage
    gw2_plus = features[features["gw"] >= 2]
    print("\nFeature coverage (GW >= 2):")
    for col in FEATURE_COLUMNS:
        non_null = gw2_plus[col].notna().sum()
        print(f"  {col}: {non_null:,} / {len(gw2_plus):,} ({non_null/len(gw2_plus):.1%})")


if __name__ == "__main__":
    main()
