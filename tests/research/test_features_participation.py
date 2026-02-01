"""
Tests for Stage 3 — Participation Feature Construction.

Validates the temporal contract and feature computation logic.
"""

import pandas as pd
import pytest

from dugout.research.pipeline.features_participation import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    compute_player_features,
    compute_participation_features,
    enforce_feature_contract,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_player_history() -> pd.DataFrame:
    """Single player with 6 GWs of data."""
    return pd.DataFrame({
        "player_id": [1] * 6,
        "gw": [1, 2, 3, 4, 5, 6],
        "minutes": [90, 0, 60, 45, 90, 0],
    })


@pytest.fixture
def multi_player_history() -> pd.DataFrame:
    """Two players with different histories."""
    return pd.DataFrame({
        "player_id": [1, 1, 1, 2, 2, 2],
        "gw": [1, 2, 3, 1, 2, 3],
        "minutes": [90, 90, 90, 0, 0, 0],
    })


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
    # Player history: GW1=90min, GW2=0min, ...
    features = compute_player_features(sample_player_history)
    gw2 = features[features["gw"] == 2].iloc[0]

    # GW 1 had 90 minutes -> played=1, played_60=1
    # So p_play_hat = 1/1 = 1.0, p60_hat = 1/1 = 1.0
    assert gw2["p_play_hat"] == 1.0
    assert gw2["p60_hat"] == 1.0


def test_gw3_uses_gw1_and_gw2(sample_player_history: pd.DataFrame) -> None:
    """GW 3 features must use GW 1-2 data only."""
    # GW1=90min (played=1, p60=1), GW2=0min (played=0, p60=0)
    features = compute_player_features(sample_player_history)
    gw3 = features[features["gw"] == 3].iloc[0]

    # p_play_hat = (1 + 0) / 2 = 0.5
    # p60_hat = (1 + 0) / 2 = 0.5
    assert gw3["p_play_hat"] == 0.5
    assert gw3["p60_hat"] == 0.5


def test_feature_never_uses_current_gw() -> None:
    """Features at GW t must never include GW t outcome."""
    # Create player who plays 90 every GW
    df = pd.DataFrame({
        "player_id": [1] * 5,
        "gw": [1, 2, 3, 4, 5],
        "minutes": [90, 90, 90, 90, 0],  # GW 5 is 0 minutes
    })
    features = compute_player_features(df)
    gw5 = features[features["gw"] == 5].iloc[0]

    # GW 5 feature should use GW 1-4 only (all 90 min)
    # p_play_hat should be 1.0, NOT affected by GW 5's 0 minutes
    assert gw5["p_play_hat"] == 1.0, "GW 5 feature incorrectly used GW 5 data"
    assert gw5["p60_hat"] == 1.0, "GW 5 feature incorrectly used GW 5 data"


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


def test_p60_leq_p_play(sample_player_history: pd.DataFrame) -> None:
    """p60_hat <= p_play_hat always (can't play 60 without playing)."""
    features = compute_player_features(sample_player_history)
    valid = features.dropna(subset=["p_play_hat", "p60_hat"])
    assert (valid["p60_hat"] <= valid["p_play_hat"] + 1e-9).all()


def test_rates_bounded_0_1(sample_player_history: pd.DataFrame) -> None:
    """Rate features are in [0, 1]."""
    features = compute_player_features(sample_player_history)
    for col in ["p_play_hat", "p60_hat", "mins_below_60_rate_5"]:
        valid = features[col].dropna()
        assert (valid >= 0).all(), f"{col} has values < 0"
        assert (valid <= 1).all(), f"{col} has values > 1"


def test_mins_std_needs_2_points() -> None:
    """mins_std_5 requires 2+ prior GWs (std of 1 point is NaN)."""
    df = pd.DataFrame({
        "player_id": [1, 1, 1],
        "gw": [1, 2, 3],
        "minutes": [90, 60, 45],
    })
    features = compute_player_features(df)

    # GW 1: NaN (no prior data)
    # GW 2: NaN (std of [90] is undefined)
    # GW 3: valid (std of [90, 60])
    assert features[features["gw"] == 1]["mins_std_5"].isna().all()
    assert features[features["gw"] == 2]["mins_std_5"].isna().all()
    assert features[features["gw"] == 3]["mins_std_5"].notna().all()


# -----------------------------------------------------------------------------
# Tests: Multi-Player
# -----------------------------------------------------------------------------


def test_features_computed_per_player(multi_player_history: pd.DataFrame) -> None:
    """Each player's features use only their own history."""
    features = compute_participation_features(multi_player_history)

    # Player 1: always plays 90 min
    p1_gw3 = features[(features["player_id"] == 1) & (features["gw"] == 3)].iloc[0]
    assert p1_gw3["p_play_hat"] == 1.0
    assert p1_gw3["p60_hat"] == 1.0

    # Player 2: never plays
    p2_gw3 = features[(features["player_id"] == 2) & (features["gw"] == 3)].iloc[0]
    assert p2_gw3["p_play_hat"] == 0.0
    assert p2_gw3["p60_hat"] == 0.0


# -----------------------------------------------------------------------------
# Tests: Contract Enforcement
# -----------------------------------------------------------------------------


def test_enforce_contract_passes(sample_player_history: pd.DataFrame) -> None:
    """Valid features pass contract enforcement."""
    features = compute_participation_features(sample_player_history)
    targets = pd.DataFrame({
        "player_id": sample_player_history["player_id"],
        "gw": sample_player_history["gw"],
    })
    enforce_feature_contract(features, targets)  # Should not raise


def test_enforce_contract_p60_gt_pplay_raises() -> None:
    """p60_hat > p_play_hat raises AssertionError."""
    # Need at least 2 GWs so GW 2 is not the player's first GW
    features = pd.DataFrame({
        "player_id": [1, 1],
        "gw": [1, 2],
        "p_play_hat": [float("nan"), 0.5],
        "p60_hat": [float("nan"), 0.8],  # Impossible: can't play 60 more often than playing
        "mins_std_5": [float("nan"), 10.0],
        "mins_below_60_rate_5": [float("nan"), 0.5],
    })
    targets = pd.DataFrame({"player_id": [1, 1], "gw": [1, 2]})

    with pytest.raises(AssertionError, match="p60_hat > p_play_hat"):
        enforce_feature_contract(features, targets)


# -----------------------------------------------------------------------------
# Tests: Determinism
# -----------------------------------------------------------------------------


def test_features_are_deterministic(sample_player_history: pd.DataFrame) -> None:
    """Same input produces identical output."""
    f1 = compute_participation_features(sample_player_history)
    f2 = compute_participation_features(sample_player_history)
    pd.testing.assert_frame_equal(f1, f2)
