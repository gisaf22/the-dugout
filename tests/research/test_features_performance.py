"""
Tests for Stage 4a — Conditional Performance Feature Construction.

Validates the temporal contract and conditional (minutes > 0) logic.
"""

import numpy as np
import pandas as pd
import pytest

from dugout.research.pipeline.features_performance import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    compute_per_90,
    compute_player_features,
    compute_performance_features,
    enforce_feature_contract,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_player_history() -> pd.DataFrame:
    """Single player with 6 GWs, some with 0 minutes."""
    return pd.DataFrame({
        "player_id": [1] * 6,
        "gw": [1, 2, 3, 4, 5, 6],
        "minutes": [90, 0, 60, 45, 90, 0],  # GW 2 and 6 = didn't play
        "total_points": [10, 0, 6, 4, 12, 0],
        "expected_goals": [0.5, 0.0, 0.3, 0.2, 0.8, 0.0],
        "expected_assists": [0.2, 0.0, 0.1, 0.1, 0.3, 0.0],
        "bonus": [3, 0, 1, 0, 3, 0],
        "ict_index": [50.0, 0.0, 30.0, 25.0, 60.0, 0.0],
    })


@pytest.fixture
def never_played_history() -> pd.DataFrame:
    """Player who never played (all minutes = 0)."""
    return pd.DataFrame({
        "player_id": [99] * 3,
        "gw": [1, 2, 3],
        "minutes": [0, 0, 0],
        "total_points": [0, 0, 0],
        "expected_goals": [0.0, 0.0, 0.0],
        "expected_assists": [0.0, 0.0, 0.0],
        "bonus": [0, 0, 0],
        "ict_index": [0.0, 0.0, 0.0],
    })


# -----------------------------------------------------------------------------
# Tests: Per-90 Computation
# -----------------------------------------------------------------------------


def test_per_90_basic() -> None:
    """Per-90 normalization works correctly."""
    stat = pd.Series([10, 5, 0])
    minutes = pd.Series([90, 45, 90])
    result = compute_per_90(stat, minutes)

    assert result[0] == 10.0  # 10/90 * 90 = 10
    assert result[1] == 10.0  # 5/45 * 90 = 10
    assert result[2] == 0.0   # 0/90 * 90 = 0


def test_per_90_zero_minutes_is_nan() -> None:
    """Per-90 is NaN when minutes = 0."""
    stat = pd.Series([5])
    minutes = pd.Series([0])
    result = compute_per_90(stat, minutes)

    assert np.isnan(result[0])


# -----------------------------------------------------------------------------
# Tests: Temporal Contract (CRITICAL)
# -----------------------------------------------------------------------------


def test_gw1_has_no_features(sample_player_history: pd.DataFrame) -> None:
    """GW 1 must have NaN features (no prior data exists)."""
    features = compute_player_features(sample_player_history)
    gw1 = features[features["gw"] == 1]

    for col in FEATURE_COLUMNS:
        assert gw1[col].isna().all(), f"GW 1 should have NaN for {col}"


def test_gw2_uses_only_gw1(sample_player_history: pd.DataFrame) -> None:
    """GW 2 features must use only GW 1 data."""
    # GW 1: 90 min, 10 pts -> points_per_90 = 10
    features = compute_player_features(sample_player_history)
    gw2 = features[features["gw"] == 2].iloc[0]

    assert gw2["points_per_90_5"] == 10.0


def test_feature_never_uses_current_gw(sample_player_history: pd.DataFrame) -> None:
    """Features at GW t must never include GW t outcome."""
    # GW 5: 90 min, 12 pts (great game)
    # GW 6 feature should NOT include GW 6's 0 points
    features = compute_player_features(sample_player_history)
    gw6 = features[features["gw"] == 6].iloc[0]

    # GW 6 uses GW 1-5 data only
    # Even though GW 6 has 0 minutes, feature should reflect prior appearances
    assert gw6["points_per_90_5"] > 0, "GW 6 feature incorrectly used GW 6 data"


def test_zero_minutes_gw_excluded_from_rolling() -> None:
    """GWs with 0 minutes should not affect rolling per-90 stats."""
    # Player: GW1=90min/10pts, GW2=0min/0pts, GW3=90min/5pts
    df = pd.DataFrame({
        "player_id": [1, 1, 1],
        "gw": [1, 2, 3],
        "minutes": [90, 0, 90],
        "total_points": [10, 0, 5],
        "expected_goals": [0.5, 0.0, 0.2],
        "expected_assists": [0.1, 0.0, 0.1],
        "bonus": [2, 0, 1],
        "ict_index": [40.0, 0.0, 30.0],
    })
    features = compute_player_features(df)

    # GW 3 feature uses GW 1-2 data
    # GW 2 had 0 minutes, so per-90 is NaN for GW 2
    # Rolling mean of [10, NaN] = 10 (NaN is skipped)
    gw3 = features[features["gw"] == 3].iloc[0]
    assert gw3["points_per_90_5"] == 10.0, "Zero-minute GW incorrectly included"


# -----------------------------------------------------------------------------
# Tests: Feature Computation
# -----------------------------------------------------------------------------


def test_output_columns(sample_player_history: pd.DataFrame) -> None:
    """Output has exactly the expected columns."""
    features = compute_player_features(sample_player_history)
    assert list(features.columns) == OUTPUT_COLUMNS


def test_row_count_matches_input(sample_player_history: pd.DataFrame) -> None:
    """One output row per input row."""
    features = compute_player_features(sample_player_history)
    assert len(features) == len(sample_player_history)


def test_per_90_features_non_negative(sample_player_history: pd.DataFrame) -> None:
    """All per-90 features are non-negative."""
    features = compute_player_features(sample_player_history)
    for col in FEATURE_COLUMNS:
        valid = features[col].dropna()
        assert (valid >= 0).all(), f"{col} has negative values"


def test_never_played_all_nan(never_played_history: pd.DataFrame) -> None:
    """Player with no appearances has all NaN features."""
    features = compute_player_features(never_played_history)
    for col in FEATURE_COLUMNS:
        assert features[col].isna().all(), f"{col} should be all NaN for never-played"


# -----------------------------------------------------------------------------
# Tests: Multi-Player
# -----------------------------------------------------------------------------


def test_features_computed_per_player() -> None:
    """Each player's features use only their own history."""
    df = pd.DataFrame({
        "player_id": [1, 1, 2, 2],
        "gw": [1, 2, 1, 2],
        "minutes": [90, 90, 45, 45],
        "total_points": [10, 10, 2, 2],  # Player 1: 10 pts/90, Player 2: 4 pts/90
        "expected_goals": [0.5, 0.5, 0.1, 0.1],
        "expected_assists": [0.2, 0.2, 0.0, 0.0],
        "bonus": [2, 2, 0, 0],
        "ict_index": [40.0, 40.0, 10.0, 10.0],
    })
    features = compute_performance_features(df)

    # GW 2 features for each player
    p1_gw2 = features[(features["player_id"] == 1) & (features["gw"] == 2)].iloc[0]
    p2_gw2 = features[(features["player_id"] == 2) & (features["gw"] == 2)].iloc[0]

    assert p1_gw2["points_per_90_5"] == 10.0  # 10 pts / 90 min * 90
    assert p2_gw2["points_per_90_5"] == 4.0   # 2 pts / 45 min * 90


# -----------------------------------------------------------------------------
# Tests: Contract Enforcement
# -----------------------------------------------------------------------------


def test_enforce_contract_passes(sample_player_history: pd.DataFrame) -> None:
    """Valid features pass contract enforcement."""
    features = compute_performance_features(sample_player_history)
    targets = pd.DataFrame({
        "player_id": sample_player_history["player_id"],
        "gw": sample_player_history["gw"],
    })
    enforce_feature_contract(features, targets)  # Should not raise


def test_enforce_contract_negative_xgi_raises() -> None:
    """Negative xGI (which can't be negative) raises AssertionError."""
    features = pd.DataFrame({
        "player_id": [1, 1],
        "gw": [1, 2],
        "points_per_90_5": [np.nan, 5.0],  # Points CAN be negative
        "xGI_per_90_5": [np.nan, -1.0],    # xGI cannot be negative
        "bonus_per_90_5": [np.nan, 1.0],
        "ict_per_90_5": [np.nan, 10.0],
    })
    targets = pd.DataFrame({"player_id": [1, 1], "gw": [1, 2]})

    with pytest.raises(AssertionError, match="negative"):
        enforce_feature_contract(features, targets)


# -----------------------------------------------------------------------------
# Tests: Determinism
# -----------------------------------------------------------------------------


def test_features_are_deterministic(sample_player_history: pd.DataFrame) -> None:
    """Same input produces identical output."""
    f1 = compute_performance_features(sample_player_history)
    f2 = compute_performance_features(sample_player_history)
    pd.testing.assert_frame_equal(f1, f2)
