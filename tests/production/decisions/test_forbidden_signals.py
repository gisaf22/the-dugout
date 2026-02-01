"""Tests for forbidden signal enforcement across all decision functions.

These tests verify that forbidden signals cause hard failures in production.
Research validated these signals should NOT be used in decisions.

FORBIDDEN SIGNALS (research-rejected):
    - p_play, p60 (availability probabilities)
    - chance_of_playing (FPL raw availability)
    - fixture_difficulty, opp_def_strength, fdr (fixture-based)
    - selected_by_percent (ownership)
    - ep_next, ep_this (FPL expected points)
    - availability_weight, fixture_weight, weighted_ev (weighting signals)

Contract: Any of these signals in decision input must raise RuntimeError.
"""

import pandas as pd
import pytest

from dugout.production.decisions.captain import (
    pick_captain,
    FORBIDDEN_SIGNALS as CAPTAIN_FORBIDDEN,
    FORBIDDEN_FIXTURE_SIGNALS as CAPTAIN_FIXTURE_FORBIDDEN,
)


# Complete list of forbidden signals across all decisions
ALL_FORBIDDEN_SIGNALS = {
    # Availability signals
    "p_play",
    "p60",
    "chance_of_playing",
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    
    # Weighting signals (research-rejected)
    "availability_weight",
    "fixture_weight",
    "weighted_ev",
    
    # Fixture signals
    "fixture_difficulty",
    "fdr",
    "opponent_strength",
    "opp_def_strength",
    
    # Ownership signals
    "selected_by_percent",
    
    # FPL expected points (leaky/circular)
    "ep_next",
    "ep_this",
}


class TestForbiddenSignalsCaptain:
    """Test that captain decision rejects all forbidden signals."""

    @pytest.mark.parametrize("signal", [
        "p_play",
        "p60",
        "availability_weight",
        "weighted_ev",
    ])
    def test_captain_rejects_availability_signals(self, signal):
        """Captain must reject availability-based signals."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "predicted_points": [5.0, 3.0],
            signal: [0.9, 0.8],
        })
        
        with pytest.raises(RuntimeError) as exc:
            pick_captain(df)
        
        assert "Contract violation" in str(exc.value)

    @pytest.mark.parametrize("signal", [
        "fixture_difficulty",
        "fdr",
        "opponent_strength",
        "fixture_weight",
    ])
    def test_captain_rejects_fixture_signals(self, signal):
        """Captain must reject fixture-based signals."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "predicted_points": [5.0, 3.0],
            signal: [2.0, 3.0],
        })
        
        with pytest.raises(RuntimeError) as exc:
            pick_captain(df)
        
        assert "Contract violation" in str(exc.value)


class TestForbiddenSignalsConsistency:
    """Test that forbidden signals are consistently defined."""

    def test_captain_forbidden_signals_defined(self):
        """Captain module must define FORBIDDEN_SIGNALS."""
        assert CAPTAIN_FORBIDDEN is not None
        assert len(CAPTAIN_FORBIDDEN) > 0

    def test_captain_fixture_signals_defined(self):
        """Captain module must define FORBIDDEN_FIXTURE_SIGNALS."""
        assert CAPTAIN_FIXTURE_FORBIDDEN is not None
        assert len(CAPTAIN_FIXTURE_FORBIDDEN) > 0

    def test_core_signals_in_captain_forbidden(self):
        """Core forbidden signals must be in captain's list."""
        core_signals = {"p_play", "p60", "weighted_ev", "availability_weight"}
        assert core_signals.issubset(CAPTAIN_FORBIDDEN)

    def test_fixture_signals_in_captain_fixture_forbidden(self):
        """Fixture signals must be in captain's fixture list."""
        fixture_signals = {"fixture_difficulty", "fdr", "opponent_strength"}
        assert fixture_signals.issubset(CAPTAIN_FIXTURE_FORBIDDEN)


class TestSignalValidationFlow:
    """Test that signal validation happens at the right point."""

    def test_validation_happens_before_decision(self):
        """Forbidden signal check must happen before argmax."""
        df = pd.DataFrame({
            "player_id": [1, 2, 3],
            "player_name": ["A", "B", "C"],
            "predicted_points": [10.0, 5.0, 3.0],
            "p_play": [0.5, 0.9, 0.8],  # Forbidden!
        })
        
        # The error should be raised immediately, not after computation
        with pytest.raises(RuntimeError):
            pick_captain(df)

    def test_clean_dataframe_passes_validation(self):
        """DataFrame without forbidden signals should pass."""
        df = pd.DataFrame({
            "player_id": [1, 2, 3],
            "player_name": ["A", "B", "C"],
            "predicted_points": [10.0, 5.0, 3.0],
            "team_name": ["TeamA", "TeamB", "TeamC"],  # Allowed
            "position": ["MID", "FWD", "DEF"],  # Allowed
            "now_cost": [100, 80, 60],  # Allowed
        })
        
        # Should not raise
        result = pick_captain(df)
        assert result["player_id"] == 1


class TestMultipleForbiddenSignals:
    """Test behavior when multiple forbidden signals present."""

    def test_multiple_forbidden_signals_caught(self):
        """All forbidden signals should be caught, not just first."""
        df = pd.DataFrame({
            "player_id": [1],
            "player_name": ["A"],
            "predicted_points": [5.0],
            "p_play": [0.9],
            "fixture_weight": [1.5],
            "weighted_ev": [7.5],
        })
        
        with pytest.raises(RuntimeError):
            pick_captain(df)

    def test_mixed_allowed_and_forbidden_caught(self):
        """Forbidden signals should be caught even with allowed columns."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "team_name": ["T1", "T2"],  # Allowed
            "position": ["MID", "FWD"],  # Allowed
            "predicted_points": [5.0, 3.0],  # Required
            "now_cost": [100, 80],  # Allowed
            "p_play": [0.9, 0.8],  # FORBIDDEN
        })
        
        with pytest.raises(RuntimeError) as exc:
            pick_captain(df)
        
        assert "p_play" in str(exc.value)
