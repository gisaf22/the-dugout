"""Free Hit Optimizer for FPL.

Optimizes a 15-player squad for a single gameweek using linear programming.
Maximizes expected points while respecting FPL constraints (budget, formation,
team limits).

Decision Rule (Frozen): Pure EV maximization
Validated by research pipeline - no variance/differential weighting.

Key Classes:
    FreeHitOptimizer - Main optimizer class
    FreeHitResult - Optimization result with squad details

Usage:
    from dugout.production.models import FreeHitOptimizer
    
    optimizer = FreeHitOptimizer(predictions_df, budget=100.0)
    result = optimizer.optimize()
    result.print_squad()
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pulp import (
    LpBinary,
    LpMaximize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    value,
)


# FreeHitMode enum removed - only basic mode supported


@dataclass
class FreeHitResult:
    """Result of Free Hit optimization.
    
    Attributes:
        starting_xi: List of starting player dicts.
        bench: List of bench player dicts (ordered 1-4).
        captain: Recommended captain.
        vice_captain: Recommended vice captain.
        total_ev: Total expected value of starting XI.
        total_cost: Total cost of squad.
        formation: Formation string (e.g., "3-5-2").
        xi_cost: Cost of starting XI only.
        fodder_cost: Cost of bench fodder (positions 2-4).
    """
    
    starting_xi: List[Dict]
    bench: List[Dict]
    captain: Dict
    vice_captain: Dict
    total_ev: float
    total_cost: float
    formation: str
    xi_cost: float = 0.0
    fodder_cost: float = 0.0
    
    def print_squad(self) -> None:
        """Print formatted squad with mode-appropriate metrics."""
        SquadFormatter(self).print()


class SquadFormatter:
    """Handles display formatting for FreeHitResult."""
    
    HEADER_LABEL = "FREE HIT OPTIMIZER"
    
    def __init__(self, result: FreeHitResult):
        self.result = result
    
    def print(self) -> None:
        """Print full formatted squad output."""
        self._print_header()
        self._print_summary()
        self._print_captain_info()
        self._print_starting_xi()
        self._print_bench()
        print("=" * 70)
    
    def _print_header(self) -> None:
        """Print optimizer header."""
        print("\n" + "=" * 70)
        print(f"🎯 {self.HEADER_LABEL}")
        print("=" * 70)
    
    def _print_summary(self) -> None:
        """Print formation, cost, and EV summary."""
        r = self.result
        print(f"\n📊 Formation: {r.formation}")
        print(f"💰 Cost: £{r.total_cost:.1f}m | XI: £{r.xi_cost:.1f}m | Fodder: £{r.fodder_cost:.1f}m")
        print(f"📈 Expected Points: {r.total_ev:.1f}")
    
    def _print_captain_info(self) -> None:
        """Print captain and vice-captain details."""
        r = self.result
        print(f"\n👑 Captain: {r.captain['name']} (EV: {r.captain['ev']:.2f})")
        print(f"🥈 Vice Captain: {r.vice_captain['name']} (EV: {r.vice_captain['ev']:.2f})")
    
    def _print_starting_xi(self) -> None:
        """Print starting XI table."""
        print("\n" + "-" * 70)
        print("STARTING XI")
        print("-" * 70)
        print(f"{'Pos':<4} {'Player':<18} {'Team':<12} {'Cost':>5} {'P(st)':>6} {'EV':>6}")
        print("-" * 70)
        
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos_players = [p for p in self.result.starting_xi if p["pos"] == pos]
            for p in sorted(pos_players, key=lambda x: -x["ev"]):
                self._print_player_row(p)
    
    def _get_captain_mark(self, player: Dict) -> str:
        """Get captain/vice-captain marker for player."""
        if player["player_id"] == self.result.captain["player_id"]:
            return " (C)"
        if player["player_id"] == self.result.vice_captain["player_id"]:
            return " (V)"
        return ""
    
    def _print_player_row(self, p: Dict) -> None:
        """Print a single player row."""
        captain_mark = self._get_captain_mark(p)
        print(
            f"{p['pos']:<4} {p['name']:<18} {p['team']:<12} "
            f"£{p['cost']:>4.1f} {p['p_start']:>5.0%} {p['ev']:>5.2f}{captain_mark}"
        )
    
    def _print_bench(self) -> None:
        """Print bench section."""
        print("\n" + "-" * 70)
        print("BENCH")
        print("-" * 70)
        for i, p in enumerate(self.result.bench, 1):
            label = "1st Sub" if i == 1 else "Fodder"
            safety = "✓" if p.get("p_start", 0) >= 0.90 else ""
            print(
                f"{i}. [{label:<6}] {p['pos']:<4} {p['name']:<18} {p['team']:<12} "
                f"£{p['cost']:>4.1f} {p['p_start']:>5.0%} {safety}"
            )


class FreeHitOptimizer:
    """Free Hit optimizer using pure EV maximization.
    
    Uses linear programming to maximize expected points while respecting
    FPL squad constraints.
    
    Example:
        >>> optimizer = FreeHitOptimizer(predictions_df, budget=100.0)
        >>> result = optimizer.optimize()
        >>> result.print_squad()
    """
    
    # Squad composition
    SQUAD_SIZE = 15
    POS_LIMITS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    MAX_PER_TEAM = 3
    
    # Formation limits (min, max starters)
    FORMATION_LIMITS = {
        "GKP": (1, 1),
        "DEF": (3, 5),
        "MID": (3, 5),
        "FWD": (1, 3),
    }
    
    # Configuration (basic mode only - pure EV maximization)
    FIRST_BENCH_WEIGHT = 0.3  # Weight for first bench slot in objective
    
    # Status handling
    UNAVAILABLE_STATUSES = {"n", "i", "s", "u"}
    DOUBTFUL_P_START_PENALTY = 0.5
    
    def __init__(
        self,
        predictions_df: pd.DataFrame,
        budget: float = 100.0,
    ) -> None:
        """Initialize optimizer.
        
        Args:
            predictions_df: DataFrame with prediction columns.
            budget: Total budget in millions.
        """
        self.df = predictions_df.copy()
        self.budget = budget
        
        self._prepare_data()
    
    def _prepare_data(self) -> None:
        """Prepare and enrich player data."""
        # Standard column mappings
        col_mappings = {
            "raw_prediction": "predicted_points",
            "player_name": "name",
            "team_name": "team",
            "now_cost": "cost",
        }
        for old, new in col_mappings.items():
            if old in self.df.columns and new not in self.df.columns:
                self.df[new] = self.df[old]
        
        if "element_type" in self.df.columns and "pos" not in self.df.columns:
            pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
            self.df["pos"] = self.df["element_type"].map(pos_map)
        
        # Default p_start if not present (must be before status handling)
        if "p_start" not in self.df.columns:
            self.df["p_start"] = 0.9  # Assume high start probability by default
        
        # Handle player status
        if "status" in self.df.columns:
            before_count = len(self.df)
            self.df = self.df[~self.df["status"].isin(self.UNAVAILABLE_STATUSES)].copy()
            filtered = before_count - len(self.df)
            if filtered > 0:
                print(f"⚠️  Excluded {filtered} unavailable players")
            
            doubtful_mask = self.df["status"] == "d"
            if doubtful_mask.any():
                self.df.loc[doubtful_mask, "p_start"] *= self.DOUBTFUL_P_START_PENALTY
                print(f"⚠️  Applied P(start) penalty to {doubtful_mask.sum()} doubtful players")
        
        # Calculate base EV (objective for optimization)
        self.df["ev"] = (
            self.df["p_start"] * self.df["predicted_points"] +
            (1 - self.df["p_start"]) * 1.0
        )
        
        # Get unique teams
        self.teams = self.df["team"].unique().tolist()
        
        # Convert to records
        self.players = self.df.to_dict("records")
        self.player_ids = [p["player_id"] for p in self.players]
    
    def optimize(self) -> Optional[FreeHitResult]:
        """Run the optimization.
        
        Returns:
            FreeHitResult with optimal squad, or None if infeasible.
        """
        prob = LpProblem("FreeHitOptimizer", LpMaximize)
        
        # Decision variables
        x = {p["player_id"]: LpVariable(f"x_{p['player_id']}", cat=LpBinary) for p in self.players}
        s = {p["player_id"]: LpVariable(f"s_{p['player_id']}", cat=LpBinary) for p in self.players}
        
        # Bench position variables
        bench_pos = {
            p["player_id"]: {k: LpVariable(f"bp_{p['player_id']}_{k}", cat=LpBinary) for k in range(4)}
            for p in self.players
        }
        
        # Bench position constraints
        for p in self.players:
            pid = p["player_id"]
            prob += lpSum(bench_pos[pid][k] for k in range(4)) == x[pid] - s[pid], f"BenchPos_{pid}"
        
        for k in range(4):
            prob += lpSum(bench_pos[pid][k] for pid in self.player_ids) == 1, f"BenchSlot_{k}"
        
        # Objective: maximize EV of starting XI + weighted first bench
        starting_ev = lpSum(s[p["player_id"]] * p["ev"] for p in self.players)
        first_bench_ev = lpSum(bench_pos[p["player_id"]][0] * p["ev"] for p in self.players)
        
        prob += starting_ev + self.FIRST_BENCH_WEIGHT * first_bench_ev, "TotalEV"
        
        # Constraints
        prob += lpSum(x[pid] for pid in self.player_ids) == self.SQUAD_SIZE, "SquadSize"
        prob += lpSum(s[pid] for pid in self.player_ids) == 11, "StartingXI"
        
        for pid in self.player_ids:
            prob += s[pid] <= x[pid], f"StartIfInSquad_{pid}"
        
        prob += lpSum(x[p["player_id"]] * p["cost"] for p in self.players) <= self.budget, "Budget"
        
        for pos, limit in self.POS_LIMITS.items():
            pos_players = [p for p in self.players if p["pos"] == pos]
            prob += lpSum(x[p["player_id"]] for p in pos_players) == limit, f"Squad_{pos}"
        
        for pos, (min_start, max_start) in self.FORMATION_LIMITS.items():
            pos_players = [p for p in self.players if p["pos"] == pos]
            prob += lpSum(s[p["player_id"]] for p in pos_players) >= min_start, f"MinStart_{pos}"
            prob += lpSum(s[p["player_id"]] for p in pos_players) <= max_start, f"MaxStart_{pos}"
        
        for team in self.teams:
            team_players = [p for p in self.players if p["team"] == team]
            if team_players:
                prob += lpSum(x[p["player_id"]] for p in team_players) <= self.MAX_PER_TEAM, f"Team_{team}"
        
        # Solve
        prob.solve()
        
        if LpStatus[prob.status] != "Optimal":
            print(f"Optimization failed: {LpStatus[prob.status]}")
            return None
        
        # Extract results
        squad = []
        for p in self.players:
            if value(x[p["player_id"]]) > 0.5:
                p_copy = p.copy()
                p_copy["is_starter"] = value(s[p["player_id"]]) > 0.5
                if not p_copy["is_starter"]:
                    for k in range(4):
                        if value(bench_pos[p["player_id"]][k]) > 0.5:
                            p_copy["bench_order"] = k
                            break
                squad.append(p_copy)
        
        starting_xi = [p for p in squad if p["is_starter"]]
        bench = sorted([p for p in squad if not p["is_starter"]], key=lambda p: p.get("bench_order", 99))
        
        formation = self._get_formation(starting_xi)
        captain, vice_captain = self._select_captain(starting_xi)
        
        total_ev = sum(p["ev"] for p in starting_xi)
        total_cost = sum(p["cost"] for p in squad)
        xi_cost = sum(p["cost"] for p in starting_xi)
        fodder_cost = sum(p["cost"] for p in bench[1:]) if len(bench) > 1 else 0
        
        return FreeHitResult(
            starting_xi=starting_xi,
            bench=bench,
            captain=captain,
            vice_captain=vice_captain,
            total_ev=total_ev,
            total_cost=total_cost,
            formation=formation,
            xi_cost=xi_cost,
            fodder_cost=fodder_cost,
        )
    
    def _get_formation(self, starting_xi: List[Dict]) -> str:
        """Get formation string from starting XI."""
        pos_counts = {"DEF": 0, "MID": 0, "FWD": 0}
        for p in starting_xi:
            if p["pos"] in pos_counts:
                pos_counts[p["pos"]] += 1
        return f"{pos_counts['DEF']}-{pos_counts['MID']}-{pos_counts['FWD']}"
    
    def _select_captain(self, starting_xi: List[Dict]) -> Tuple[Dict, Dict]:
        """Select captain and vice captain by expected value (EV).
        
        Frozen policy: argmax(predicted_points)
        """
        sorted_by_ev = sorted(starting_xi, key=lambda p: -p.get("ev", 0))
        
        captain = sorted_by_ev[0]
        
        # Vice captain: high EV with high P(start) for safety
        safe_players = [p for p in sorted_by_ev if p.get("p_start", 0) >= 0.95 and p != captain]
        if safe_players:
            vice_captain = safe_players[0]
        else:
            vice_captain = sorted_by_ev[1] if len(sorted_by_ev) > 1 else captain
        
        return captain, vice_captain


def run_free_hit_optimizer(
    predictions_df: pd.DataFrame,
    budget: float = 100.0,
) -> Optional[FreeHitResult]:
    """Convenience function to run the Free Hit optimizer.
    
    Args:
        predictions_df: DataFrame with prediction columns.
        budget: Budget in millions.
    
    Returns:
        FreeHitResult or None if failed.
    """
    optimizer = FreeHitOptimizer(predictions_df, budget=budget)
    return optimizer.optimize()
