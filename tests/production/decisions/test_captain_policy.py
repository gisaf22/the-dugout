"""Tests for research-validated captain selection policy.

These tests verify that production captain selection adheres to
the frozen decision rule: argmax(predicted_points).

The research pipeline (Stage 6) explicitly rejected:
- Availability weighting (p_play × points)
- p60 threshold filtering

See: docs/DECISION_CONTRACT_LAYER.md
"""

import pandas as pd
import pytest

from dugout.production.models.captain import CaptainPicker


@pytest.fixture
def sample_predictions():
    """Create sample prediction data for testing."""
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5],
        "player_name": ["Player A", "Player B", "Player C", "Player D", "Player E"],
        "team_name": ["Team 1", "Team 2", "Team 3", "Team 4", "Team 5"],
        "predicted_points": [8.0, 7.5, 7.0, 6.5, 6.0],
        "p_start": [0.60, 0.99, 0.95, 0.85, 0.99],  # Player A has LOW p_start
    })


class TestFrozenCaptainPolicy:
    """Tests ensuring captain selection follows frozen rule: argmax(predicted_points)."""

    def test_captain_uses_expected_points_not_availability_weighted(
        self, sample_predictions
    ):
        """Captain should be argmax(predicted_points), NOT argmax(p_start × predicted_points).
        
        Research finding (Stage 6): Availability weighting increases regret.
        
        In this test:
        - Player A: predicted_points=8.0, p_start=0.60 → weighted=4.8
        - Player B: predicted_points=7.5, p_start=0.99 → weighted=7.43
        
        If using availability weighting, Player B would be selected (higher weighted).
        Frozen rule selects Player A (highest predicted_points).
        """
        picker = CaptainPicker(sample_predictions)
        pick = picker.get_recommendation()
        
        # Player A has highest predicted_points (8.0), despite low p_start (0.60)
        assert pick.player_id == 1, (
            "Captain should be Player A (highest predicted_points), "
            "not Player B (which would win under availability weighting)"
        )
        assert pick.player_name == "Player A"

    def test_captain_selection_does_not_filter_by_p_start(self, sample_predictions):
        """All players should be candidates regardless of p_start.
        
        Research finding: p60/p_start filtering was rejected - it reduces
        candidate pool without improving regret.
        """
        picker = CaptainPicker(sample_predictions)
        pick = picker.get_recommendation()
        
        # Player A has p_start=0.60, well below typical "starter" thresholds
        # Frozen rule does NOT filter these out
        assert pick.p_start == 0.60

    def test_ranking_order_matches_expected_points(self, sample_predictions):
        """Captain rankings should match predicted_points order exactly."""
        picker = CaptainPicker(sample_predictions)
        
        # Internal df should be ranked by predicted_points
        sorted_by_pts = picker.df.nlargest(5, "predicted_points")
        expected_order = [1, 2, 3, 4, 5]  # Player IDs in predicted_points order
        
        assert sorted_by_pts["player_id"].tolist() == expected_order

    def test_get_recommendations_backward_compatibility(self, sample_predictions):
        """get_recommendations returns list with single pick for compatibility."""
        picker = CaptainPicker(sample_predictions)
        picks = picker.get_recommendations()
        
        assert len(picks) == 1
        assert picks[0].player_id == 1  # argmax(predicted_points)

    def test_empty_squad_returns_none(self):
        """Empty candidate set returns None."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "team_name": ["T1", "T2"],
            "predicted_points": [5.0, 4.0],
            "p_start": [0.9, 0.8],
        })
        picker = CaptainPicker(df)
        
        # Filter to non-existent squad
        pick = picker.get_recommendation(squad_ids=[999, 998])
        assert pick is None
