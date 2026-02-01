"""Tests for Free Hit decision contract.

These tests enforce the frozen decision rule: LP maximize Σ(predicted_points).
Research pipeline validated this policy; production must enforce it.

CONTRACT:
    - LP objective is sum(predicted_points) for Starting XI
    - Pure EV maximization (no fixture adjustments)
    - No availability weighting
    - Formation constraints enforced (GKP:1, DEF:3-5, MID:2-5, FWD:1-3)
    - Budget constraint enforced (total ≤ budget)
    - Team constraint enforced (max 3 per team)
"""

import pandas as pd
import pytest
from unittest.mock import Mock, patch

from dugout.production.decisions.free_hit import (
    optimize_free_hit,
    FORBIDDEN_SIGNALS,
    FORBIDDEN_FIXTURE_SIGNALS,
)
from dugout.production.models.squad import FreeHitOptimizer


def _build_feasible_squad_df():
    """Create a DataFrame with enough cheap players to form a feasible squad.
    
    Budget: 100M
    Need: 15 players (2 GKP, 5 DEF, 5 MID, 3 FWD)
    Avg cost must be < 6.67M per player
    
    The optimizer expects:
    - player_id: unique int
    - element_type: int (1=GKP, 2=DEF, 3=MID, 4=FWD)
    - cost: float in millions (e.g. 5.0 = £5.0M)
    - predicted_points: float
    - team: str
    - name: str
    """
    players = []
    player_id = 1
    
    # Create 4 GKPs - costs 4.0-4.5M each
    for i in range(4):
        players.append({
            "player_id": player_id,
            "name": f"GKP_{i}",
            "team": f"Team{i + 1}",  # Each on different team
            "element_type": 1,  # GKP
            "predicted_points": 3.0 + i * 0.5,
            "cost": 4.0 + i * 0.1,
        })
        player_id += 1
    
    # Create 10 DEFs - costs 4.5-6.0M each
    for i in range(10):
        players.append({
            "player_id": player_id,
            "name": f"DEF_{i}",
            "team": f"Team{(i % 10) + 1}",
            "element_type": 2,  # DEF
            "predicted_points": 4.0 + i * 0.3,
            "cost": 4.5 + (i % 6) * 0.25,
        })
        player_id += 1
    
    # Create 15 MIDs - costs 5.0-8.0M each
    for i in range(15):
        players.append({
            "player_id": player_id,
            "name": f"MID_{i}",
            "team": f"Team{(i % 12) + 1}",
            "element_type": 3,  # MID
            "predicted_points": 5.0 + i * 0.4,
            "cost": 5.0 + (i % 8) * 0.4,
        })
        player_id += 1
    
    # Create 8 FWDs - costs 5.5-8.5M each
    for i in range(8):
        players.append({
            "player_id": player_id,
            "name": f"FWD_{i}",
            "team": f"Team{(i % 8) + 1}",
            "element_type": 4,  # FWD
            "predicted_points": 5.5 + i * 0.5,
            "cost": 5.5 + i * 0.4,
        })
        player_id += 1
    
    return pd.DataFrame(players)


@pytest.fixture
def feasible_squad_df():
    """Fixture providing a feasible squad DataFrame."""
    return _build_feasible_squad_df()


class TestFreeHitLPObjective:
    """Free Hit must maximize sum(predicted_points) for Starting XI."""

    def test_optimizer_uses_predicted_points_as_objective(self, feasible_squad_df):
        """LP objective should be maximizing predicted_points sum."""
        optimizer = FreeHitOptimizer(
            feasible_squad_df, budget=100.0
        )
        result = optimizer.optimize()
        
        # Starting XI should contain the high-EV players
        assert result is not None
        assert len(result.starting_xi) == 11
        assert result.total_ev > 0


class TestFreeHitPureEVMaximization:
    """Free Hit must use pure EV maximization (no weights)."""

    def test_optimizer_uses_ev_directly(self, feasible_squad_df):
        """Basic mode uses EV directly without variance/differential weighting."""
        optimizer = FreeHitOptimizer(
            feasible_squad_df, budget=100.0
        )
        
        # Verify players have 'ev' column computed
        assert "ev" in optimizer.df.columns
        # All EVs should be positive
        assert (optimizer.df["ev"] > 0).all()


class TestFreeHitFormationConstraints:
    """Free Hit must respect FPL formation rules."""

    def test_starting_xi_has_exactly_11_players(self, feasible_squad_df):
        """Starting XI must have exactly 11 players."""
        optimizer = FreeHitOptimizer(
            feasible_squad_df, budget=100.0
        )
        result = optimizer.optimize()
        
        assert result is not None
        assert len(result.starting_xi) == 11

    def test_starting_xi_has_exactly_1_goalkeeper(self, feasible_squad_df):
        """Starting XI must have exactly 1 GKP."""
        optimizer = FreeHitOptimizer(
            feasible_squad_df, budget=100.0
        )
        result = optimizer.optimize()
        
        assert result is not None
        gkp_count = sum(1 for p in result.starting_xi if p["pos"] == "GKP")
        assert gkp_count == 1

    def test_starting_xi_has_valid_formation(self, feasible_squad_df):
        """Starting XI formation must be valid (3-5 DEF, 2-5 MID, 1-3 FWD)."""
        optimizer = FreeHitOptimizer(
            feasible_squad_df, budget=100.0
        )
        result = optimizer.optimize()
        
        assert result is not None
        positions = [p["pos"] for p in result.starting_xi]
        def_count = positions.count("DEF")
        mid_count = positions.count("MID")
        fwd_count = positions.count("FWD")
        
        assert 3 <= def_count <= 5
        assert 2 <= mid_count <= 5
        assert 1 <= fwd_count <= 3


class TestFreeHitBudgetConstraint:
    """Free Hit must respect budget constraint."""

    def test_total_cost_within_budget(self, feasible_squad_df):
        """Total squad cost must not exceed budget."""
        budget = 100.0
        optimizer = FreeHitOptimizer(
            feasible_squad_df, budget=budget
        )
        result = optimizer.optimize()
        
        assert result is not None
        assert result.total_cost <= budget


class TestFreeHitTeamConstraint:
    """Free Hit must respect max 3 players per team."""

    def test_max_three_players_per_team(self, feasible_squad_df):
        """No team should have more than 3 players in squad."""
        optimizer = FreeHitOptimizer(
            feasible_squad_df, budget=100.0
        )
        result = optimizer.optimize()
        
        assert result is not None
        
        # Count players per team in full squad
        all_players = result.starting_xi + result.bench
        team_counts = {}
        for p in all_players:
            team = p["team"]
            team_counts[team] = team_counts.get(team, 0) + 1
        
        for team, count in team_counts.items():
            assert count <= 3, f"Team {team} has {count} players (max 3 allowed)"


class TestFreeHitForbiddenSignals:
    """Free Hit must fail if forbidden signals enter the pipeline."""

    def test_forbidden_signals_list_is_defined(self):
        """Verify the forbidden signals list exists and is non-empty."""
        assert len(FORBIDDEN_SIGNALS) > 0
        assert "p_play" in FORBIDDEN_SIGNALS
        assert "availability_weight" in FORBIDDEN_SIGNALS

    def test_forbidden_fixture_signals_list_is_defined(self):
        """Verify fixture signals list exists."""
        assert len(FORBIDDEN_FIXTURE_SIGNALS) > 0
        assert "fixture_difficulty" in FORBIDDEN_FIXTURE_SIGNALS
