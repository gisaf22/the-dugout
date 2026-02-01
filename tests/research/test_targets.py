"""
Tests for Stage 2 — Target Construction.

These tests validate the contract enforcement logic without
requiring the full database.
"""

import pandas as pd
import pytest

from dugout.research.pipeline.targets import (
    BINARY_TARGETS,
    EXPECTED_COLUMNS,
    construct_targets,
    enforce_prediction_contract,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_gw_outcomes() -> pd.DataFrame:
    """Minimal valid gameweek outcomes."""
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2],
            "gw": [1, 2, 1, 2],
            "total_points": [2, 10, 0, 15],
            "minutes": [90, 60, 0, 45],
        }
    )


# -----------------------------------------------------------------------------
# Tests: construct_targets
# -----------------------------------------------------------------------------


def test_construct_targets_columns(sample_gw_outcomes: pd.DataFrame) -> None:
    """Output has exactly the expected columns."""
    targets = construct_targets(sample_gw_outcomes)
    assert list(targets.columns) == EXPECTED_COLUMNS


def test_construct_targets_row_count(sample_gw_outcomes: pd.DataFrame) -> None:
    """One output row per input row."""
    targets = construct_targets(sample_gw_outcomes)
    assert len(targets) == len(sample_gw_outcomes)


def test_construct_targets_y_points(sample_gw_outcomes: pd.DataFrame) -> None:
    """y_points equals total_points."""
    targets = construct_targets(sample_gw_outcomes)
    assert (targets["y_points"] == sample_gw_outcomes["total_points"]).all()


def test_construct_targets_y_play(sample_gw_outcomes: pd.DataFrame) -> None:
    """y_play is 1 iff minutes > 0."""
    targets = construct_targets(sample_gw_outcomes)
    expected = (sample_gw_outcomes["minutes"] > 0).astype(int)
    assert (targets["y_play"] == expected).all()


def test_construct_targets_y_60(sample_gw_outcomes: pd.DataFrame) -> None:
    """y_60 is 1 iff minutes >= 60."""
    targets = construct_targets(sample_gw_outcomes)
    expected = (sample_gw_outcomes["minutes"] >= 60).astype(int)
    assert (targets["y_60"] == expected).all()


def test_construct_targets_y_haul(sample_gw_outcomes: pd.DataFrame) -> None:
    """y_haul is 1 iff total_points >= 10."""
    targets = construct_targets(sample_gw_outcomes)
    expected = (sample_gw_outcomes["total_points"] >= 10).astype(int)
    assert (targets["y_haul"] == expected).all()


# -----------------------------------------------------------------------------
# Tests: enforce_prediction_contract
# -----------------------------------------------------------------------------


def test_enforce_prediction_contract_passes(sample_gw_outcomes: pd.DataFrame) -> None:
    """Valid targets pass contract enforcement."""
    targets = construct_targets(sample_gw_outcomes)
    enforce_prediction_contract(targets)  # Should not raise


def test_enforce_contract_duplicate_raises() -> None:
    """Duplicate (player_id, gw) raises AssertionError."""
    targets = pd.DataFrame(
        {
            "player_id": [1, 1],  # Same player, same GW
            "gw": [1, 1],
            "y_points": [2, 3],
            "y_play": [1, 1],
            "y_60": [1, 1],
            "y_haul": [0, 0],
        }
    )
    with pytest.raises(AssertionError, match="duplicate"):
        enforce_prediction_contract(targets)


def test_enforce_contract_invalid_binary_raises() -> None:
    """Binary target with value outside {0,1} raises AssertionError."""
    targets = pd.DataFrame(
        {
            "player_id": [1],
            "gw": [1],
            "y_points": [2],
            "y_play": [2],  # Invalid: not in {0,1}
            "y_60": [0],
            "y_haul": [0],
        }
    )
    with pytest.raises(AssertionError, match="y_play"):
        enforce_prediction_contract(targets)


def test_enforce_contract_y60_without_yplay_raises() -> None:
    """y_60=1 with y_play=0 raises AssertionError."""
    targets = pd.DataFrame(
        {
            "player_id": [1],
            "gw": [1],
            "y_points": [2],
            "y_play": [0],  # Didn't play
            "y_60": [1],  # But played 60+? Impossible.
            "y_haul": [0],
        }
    )
    with pytest.raises(AssertionError, match="y_60=1 but y_play=0"):
        enforce_prediction_contract(targets)


def test_enforce_contract_yhaul_without_yplay_raises() -> None:
    """y_haul=1 with y_play=0 raises AssertionError."""
    targets = pd.DataFrame(
        {
            "player_id": [1],
            "gw": [1],
            "y_points": [12],
            "y_play": [0],  # Didn't play
            "y_60": [0],
            "y_haul": [1],  # But hauled? Impossible.
        }
    )
    with pytest.raises(AssertionError, match="y_haul=1 but y_play=0"):
        enforce_prediction_contract(targets)


def test_enforce_contract_missing_column_raises() -> None:
    """Missing column raises AssertionError."""
    targets = pd.DataFrame(
        {
            "player_id": [1],
            "gw": [1],
            "y_points": [2],
            # Missing: y_play, y_60, y_haul
        }
    )
    with pytest.raises(AssertionError, match="Column mismatch"):
        enforce_prediction_contract(targets)


# -----------------------------------------------------------------------------
# Tests: Determinism
# -----------------------------------------------------------------------------


def test_construct_targets_is_deterministic(sample_gw_outcomes: pd.DataFrame) -> None:
    """Same input produces identical output."""
    targets1 = construct_targets(sample_gw_outcomes)
    targets2 = construct_targets(sample_gw_outcomes)
    pd.testing.assert_frame_equal(targets1, targets2)
