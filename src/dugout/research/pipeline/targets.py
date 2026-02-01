"""
Stage 2 — Target Construction (Contract Enforcement)

This module constructs the ground-truth supervised learning targets
at the player_id × gw level, enforcing the prediction contract.

Output: storage/datasets/targets.csv

Schema:
    player_id   int     Player identifier
    gw          int     Target gameweek (t)
    y_points    int     Points scored in GW t
    y_play      int     1 if minutes > 0 in GW t
    y_60        int     1 if minutes >= 60 in GW t
    y_haul      int     1 if points >= 10 in GW t

Construction Rules:
    - One row per (player_id, gw)
    - Targets derived ONLY from GW t outcomes
    - No joins to GW t-1
    - No feature columns
    - No snapshot reconstruction
    - No inference

Guarantees:
    - Deterministic
    - Idempotent (same input -> same output)
    - Free of temporal leakage
"""

import sqlite3
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

EXPECTED_COLUMNS = ["player_id", "gw", "y_points", "y_play", "y_60", "y_haul"]
BINARY_TARGETS = ["y_play", "y_60", "y_haul"]


# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------


def load_gameweek_outcomes(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load raw gameweek outcomes from the database.

    Returns only the columns needed for target construction:
    - element_id (renamed to player_id)
    - round (renamed to gw)
    - total_points
    - minutes

    No filtering, no aggregation, no lagging.
    """
    query = """
    SELECT 
        element_id AS player_id,
        round AS gw,
        total_points,
        minutes
    FROM gameweeks
    ORDER BY element_id, round
    """
    return pd.read_sql(query, conn)


def construct_targets(gw_outcomes: pd.DataFrame) -> pd.DataFrame:
    """
    Construct target columns from raw gameweek outcomes.

    Each row represents what happened in GW t — not what was known before GW t.

    Args:
        gw_outcomes: DataFrame with columns [player_id, gw, total_points, minutes]

    Returns:
        DataFrame with columns [player_id, gw, y_points, y_play, y_60, y_haul]
    """
    targets = pd.DataFrame(
        {
            "player_id": gw_outcomes["player_id"],
            "gw": gw_outcomes["gw"],
            "y_points": gw_outcomes["total_points"],
            "y_play": (gw_outcomes["minutes"] > 0).astype(int),
            "y_60": (gw_outcomes["minutes"] >= 60).astype(int),
            "y_haul": (gw_outcomes["total_points"] >= 10).astype(int),
        }
    )
    return targets


# -----------------------------------------------------------------------------
# Contract Enforcement
# -----------------------------------------------------------------------------


def enforce_prediction_contract(targets: pd.DataFrame) -> None:
    """
    Enforce the prediction contract on the target DataFrame.

    Validates that targets conform to the Stage 2 contract:
    - Schema matches exactly
    - No missing values
    - (player_id, gw) uniqueness
    - Binary targets in {0, 1}
    - Logical consistency (y_60 → y_play, y_haul → y_play)

    Raises AssertionError if any check fails.
    """
    # Check 1: Expected columns present
    assert list(targets.columns) == EXPECTED_COLUMNS, (
        f"Column mismatch. Expected {EXPECTED_COLUMNS}, got {list(targets.columns)}"
    )

    # Check 2: No missing values
    missing = targets.isnull().sum().sum()
    assert missing == 0, f"Found {missing} missing values"

    # Check 3: (player_id, gw) uniqueness
    duplicates = targets.duplicated(subset=["player_id", "gw"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (player_id, gw) pairs"

    # Check 4: Binary targets are in {0, 1}
    for col in BINARY_TARGETS:
        unique_vals = set(targets[col].unique())
        assert unique_vals <= {0, 1}, (
            f"Column {col} has values outside {{0,1}}: {unique_vals}"
        )

    # Check 5: Logical consistency — y_60 implies y_play
    invalid_60 = ((targets["y_60"] == 1) & (targets["y_play"] == 0)).sum()
    assert invalid_60 == 0, f"Found {invalid_60} rows where y_60=1 but y_play=0"

    # Check 6: Logical consistency — y_haul implies y_play
    invalid_haul = ((targets["y_haul"] == 1) & (targets["y_play"] == 0)).sum()
    assert invalid_haul == 0, f"Found {invalid_haul} rows where y_haul=1 but y_play=0"

    # Check 7: GW values are positive integers
    assert (targets["gw"] > 0).all(), "GW values must be positive integers"


# -----------------------------------------------------------------------------
# Pipeline Entry Point
# -----------------------------------------------------------------------------


def build_targets(db_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Stage 2 pipeline entry point.

    Loads raw gameweek data, constructs targets, validates, and saves.

    Args:
        db_path: Path to the SQLite database
        output_path: Path to write targets.csv

    Returns:
        The validated targets DataFrame
    """
    # Load raw outcomes
    conn = sqlite3.connect(db_path)
    try:
        gw_outcomes = load_gameweek_outcomes(conn)
    finally:
        conn.close()

    # Construct targets
    targets = construct_targets(gw_outcomes)

    # Enforce contract
    enforce_prediction_contract(targets)

    # Save (idempotent — overwrites if exists)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_path, index=False)

    return targets


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for Stage 2."""
    import os

    # Resolve paths: targets.py -> pipeline -> dugout -> src -> project_root
    project_root = Path(__file__).resolve().parents[4]
    db_path = Path(
        os.environ.get(
            "DUGOUT_DB_PATH",
            project_root / "storage" / "fpl_2025_26.sqlite",
        )
    )
    output_path = project_root / "storage" / "research" / "datasets" / "targets.csv"

    print(f"Stage 2 — Target Construction")
    print(f"  Database: {db_path}")
    print(f"  Output:   {output_path}")

    targets = build_targets(db_path, output_path)

    print(f"\nCompleted.")
    print(f"  Rows:    {len(targets):,}")
    print(f"  Players: {targets['player_id'].nunique():,}")
    print(f"  GWs:     {targets['gw'].min()} to {targets['gw'].max()}")


if __name__ == "__main__":
    main()
