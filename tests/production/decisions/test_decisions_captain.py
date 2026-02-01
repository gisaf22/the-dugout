"""Tests for captain decision contract.

These tests enforce the frozen decision rule: argmax(predicted_points).
Research pipeline validated this policy; production must enforce it.

CONTRACT:
    - Captain uses ONLY predicted_points
    - Forbidden signals cause RuntimeError
    - No availability weighting
    - No fixture-based adjustments in decision
"""

import pandas as pd
import pytest

from dugout.production.decisions.captain import (
    pick_captain,
    FORBIDDEN_SIGNALS,
    FORBIDDEN_FIXTURE_SIGNALS,
)


class TestCaptainUsesOnlyPredictedPoints:
    """Captain selection must use argmax(predicted_points) only."""

    def test_captain_uses_only_predicted_points(self):
        """Captain should be the player with highest predicted_points."""
        df = pd.DataFrame({
            "player_id": [1, 2, 3],
            "player_name": ["Low", "Mid", "High"],
            "predicted_points": [3.0, 5.0, 8.0],
        })
        
        captain = pick_captain(df)
        
        assert captain["player_id"] == 3
        assert captain["player_name"] == "High"
        assert captain["predicted_points"] == 8.0

    def test_captain_ignores_other_columns(self):
        """Captain selection should ignore non-predicted_points columns."""
        df = pd.DataFrame({
            "player_id": [1, 2, 3],
            "player_name": ["A", "B", "C"],
            "predicted_points": [10.0, 5.0, 2.0],
            "ownership": [50.0, 10.0, 5.0],  # Ignored
            "minutes": [90, 45, 90],  # Ignored
            "form": [1.0, 10.0, 5.0],  # Ignored
        })
        
        captain = pick_captain(df)
        
        # Player A wins on predicted_points despite low form/ownership metrics
        assert captain["player_id"] == 1

    def test_captain_does_not_use_availability_weighting(self):
        """Captain should NOT be p_start × predicted_points.
        
        Research finding: Availability weighting increases regret.
        
        Player 1: predicted_points=10.0, p_start=0.50 → weighted=5.0
        Player 2: predicted_points=6.0, p_start=0.99 → weighted=5.94
        
        Wrong behavior: Player 2 selected (higher weighted)
        Correct behavior: Player 1 selected (higher predicted_points)
        """
        df = pd.DataFrame({
            "player_id": [1, 2],
            "player_name": ["Risky", "Safe"],
            "predicted_points": [10.0, 6.0],
            "p_start": [0.50, 0.99],  # Should be ignored
        })
        
        captain = pick_captain(df)
        
        assert captain["player_id"] == 1, (
            "Captain should be Player 1 (highest predicted_points), "
            "not Player 2 (which would win under availability weighting)"
        )

    def test_captain_ranking_matches_predicted_points_order(self):
        """Top-N candidates should be ordered by predicted_points descending."""
        df = pd.DataFrame({
            "player_id": [1, 2, 3, 4, 5],
            "player_name": ["A", "B", "C", "D", "E"],
            "predicted_points": [3.0, 8.0, 5.0, 10.0, 1.0],
        })
        
        # Get captain (should be player_id=4 with 10.0)
        captain = pick_captain(df)
        assert captain["player_id"] == 4
        
        # Verify order by sorting
        sorted_df = df.nlargest(5, "predicted_points")
        expected_order = [4, 2, 3, 1, 5]  # 10.0, 8.0, 5.0, 3.0, 1.0
        assert sorted_df["player_id"].tolist() == expected_order


class TestCaptainRejectsForbiddenSignals:
    """Captain must raise RuntimeError if forbidden signals are present."""

    @pytest.mark.parametrize("forbidden_col", list(FORBIDDEN_SIGNALS))
    def test_captain_rejects_forbidden_signal(self, forbidden_col):
        """Each forbidden signal should cause RuntimeError."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "predicted_points": [5.0, 3.0],
            forbidden_col: [0.8, 0.9],  # Forbidden!
        })
        
        with pytest.raises(RuntimeError) as exc_info:
            pick_captain(df)
        
        assert "Contract violation" in str(exc_info.value)
        assert forbidden_col in str(exc_info.value)

    @pytest.mark.parametrize("fixture_col", list(FORBIDDEN_FIXTURE_SIGNALS))
    def test_captain_rejects_fixture_columns(self, fixture_col):
        """Fixture-based columns should cause RuntimeError."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "predicted_points": [5.0, 3.0],
            fixture_col: [1.5, 2.0],  # Forbidden!
        })
        
        with pytest.raises(RuntimeError) as exc_info:
            pick_captain(df)
        
        assert "Contract violation" in str(exc_info.value)

    def test_captain_rejects_multiple_forbidden_signals(self):
        """Multiple forbidden signals should all be caught."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "predicted_points": [5.0, 3.0],
            "p_play": [0.8, 0.9],
            "fixture_weight": [1.0, 1.5],
        })
        
        with pytest.raises(RuntimeError):
            pick_captain(df)


class TestCaptainRequiresPredictedPoints:
    """Captain must have predicted_points column."""

    def test_captain_requires_predicted_points_column(self):
        """Missing predicted_points should cause RuntimeError."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "points": [5.0, 3.0],  # Wrong column name!
        })
        
        with pytest.raises(RuntimeError) as exc_info:
            pick_captain(df)
        
        assert "predicted_points" in str(exc_info.value)

    def test_captain_with_empty_df(self):
        """Empty DataFrame should raise appropriate error."""
        df = pd.DataFrame(columns=["player_id", "player_name", "predicted_points"])
        
        # Should raise an error (either KeyError, IndexError, or RuntimeError)
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            pick_captain(df)
