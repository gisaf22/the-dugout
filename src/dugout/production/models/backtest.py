"""Captain backtesting module.

Backtests captain selection using frozen rule: argmax(predicted_points)
against historical actual outcomes.

⚠️  DIAGNOSTIC EXCEPTION: This module contains lgb.train() calls.
    This is intentional and NOT a violation of the canonical training path.
    
    WHY: Walk-forward captain backtest requires training a fresh model for each
    test gameweek (train on GW 1..t-1, predict GW t). This is fundamentally
    different from production training which trains once and deploys.
    
    RULE: Models trained here are NEVER saved or used for production.
    Production models MUST be trained via dugout.production.pipeline.trainer.

Key Metrics:
    - Captain hit rate: How often captain is top scorer in squad
    - Points vs optimal: Points lost by not picking the best player
    - Squad recall: How often global best was in our squad

Design:
    Uses walk-forward validation: For each test GW, train model on
    prior GWs, generate predictions, pick captain, compare to actual.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal
import numpy as np
import pandas as pd
import lightgbm as lgb

from dugout.production.features.definitions import FEATURE_COLUMNS


@dataclass
class CaptainGWResult:
    """Result from captain selection for a single gameweek."""
    
    gw: int
    
    # Captain selected
    captain_id: int
    captain_name: str
    captain_predicted: float
    captain_actual: float
    captain_doubled: float
    was_haul: bool  # 10+ points
    
    # Optimal captain (best in squad)
    optimal_id: int
    optimal_name: str
    optimal_actual: float
    optimal_doubled: float
    
    # Global optimal (best in entire GW, regardless of squad)
    global_optimal_id: int
    global_optimal_name: str
    global_optimal_actual: float
    global_optimal_in_squad: bool  # Was global best in our predicted squad?
    
    # Regret
    regret: float  # optimal_doubled - captain_doubled
    was_optimal: bool  # captain was the best pick in squad


@dataclass
class CaptainBacktestSummary:
    """Summary of captain backtest across all gameweeks."""
    
    n_gameweeks: int
    
    # Points
    total_doubled_points: float
    avg_doubled_points: float
    optimal_doubled_points: float
    
    # Hit rate (captain was best in squad)
    hit_count: int
    hit_rate: float
    
    # Haul rate (captain scored 10+)
    haul_count: int
    haul_rate: float
    
    # Squad recall (global best was in squad)
    squad_recall_count: int
    squad_recall_rate: float
    
    # Regret
    total_regret: float
    avg_regret: float
    
    # Per-GW results for analysis
    gw_results: List[CaptainGWResult] = field(default_factory=list)
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"\n{'=' * 50}",
            f"CAPTAIN BACKTEST: argmax(predicted_points)",
            f"{'=' * 50}",
            f"Gameweeks: {self.n_gameweeks}",
            f"",
            f"POINTS:",
            f"  Total (doubled): {self.total_doubled_points:.0f}",
            f"  Avg per GW:      {self.avg_doubled_points:.2f}",
            f"  Optimal total:   {self.optimal_doubled_points:.0f}",
            f"",
            f"HIT RATE (captain = top in squad):",
            f"  {self.hit_count}/{self.n_gameweeks} = {self.hit_rate:.1%}",
            f"",
            f"HAUL RATE (captain 10+):",
            f"  {self.haul_count}/{self.n_gameweeks} = {self.haul_rate:.1%}",
            f"",
            f"SQUAD RECALL (global best in squad):",
            f"  {self.squad_recall_count}/{self.n_gameweeks} = {self.squad_recall_rate:.1%}",
            f"",
            f"REGRET:",
            f"  Total: {self.total_regret:.0f} pts",
            f"  Avg:   {self.avg_regret:.2f} pts/GW",
            f"{'=' * 50}",
        ]
        return "\n".join(lines)


class CaptainBacktester:
    """
    Backtests captain selection using frozen rule: argmax(predicted_points).
    
    For each test gameweek:
    1. Train model on all prior GWs
    2. Generate predictions for test GW players
    3. Select captain as argmax(predicted_points)
    4. Compare captain's actual points to optimal
    """
    
    def __init__(
        self,
        min_train_gws: int = 5,
        target_column: str = "total_points",
    ):
        """
        Args:
            min_train_gws: Minimum GWs needed before first prediction
            target_column: Column with actual points
        """
        self.min_train_gws = min_train_gws
        self.target_column = target_column
        
        # LightGBM params - deterministic (no bagging/subsampling)
        self.lgb_params = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": 0.07,
            "num_leaves": 8,
            "min_data_in_leaf": 150,
            "seed": 42,
        }
    
    def run(
        self,
        df: pd.DataFrame,
        squad_size: int = 15,
        verbose: bool = True,
    ) -> CaptainBacktestSummary:
        """
        Run captain backtest using frozen rule: argmax(predicted_points).
        
        Args:
            df: Full feature DataFrame with all GWs
            squad_size: Top N players per GW to consider as squad
            verbose: Print progress
            
        Returns:
            CaptainBacktestSummary with results
        """
        all_gws = sorted(df["gw"].unique())
        start_gw = all_gws[self.min_train_gws]
        test_gws = [gw for gw in all_gws if gw >= start_gw]
        
        if verbose:
            print(f"Running captain backtest: argmax(predicted_points)")
            print(f"  Test GWs: {test_gws[0]}-{test_gws[-1]} ({len(test_gws)} weeks)")
        
        results = []
        
        for test_gw in test_gws:
            # Split train/test
            train_df = df[df["gw"] < test_gw]
            test_df = df[df["gw"] == test_gw].copy()
            
            if len(train_df) < 100 or len(test_df) < squad_size:
                continue
            
            # Train model
            model = self._train_model(train_df)
            
            # Generate predictions
            X_test = test_df[FEATURE_COLUMNS].values
            test_df["predicted"] = self._predict(model, X_test)
            
            # Simulate squad: top N by predicted points
            squad = test_df.nlargest(squad_size, "predicted")
            squad_ids = set(squad["player_id"].values)
            
            # Frozen rule: argmax(predicted_points)
            captain = squad.nlargest(1, "predicted").iloc[0]
            
            # Find optimal captain (best actual in squad)
            optimal = squad.nlargest(1, self.target_column).iloc[0]
            
            # Find global optimal (best in entire GW)
            global_opt = test_df.nlargest(1, self.target_column).iloc[0]
            global_opt_in_squad = int(global_opt["player_id"]) in squad_ids
            
            # Record result
            captain_actual = float(captain[self.target_column])
            optimal_actual = float(optimal[self.target_column])
            
            result = CaptainGWResult(
                gw=test_gw,
                captain_id=int(captain["player_id"]),
                captain_name=str(captain.get("player_name", "Unknown")),
                captain_predicted=float(captain["predicted"]),
                captain_actual=captain_actual,
                captain_doubled=captain_actual * 2,
                was_haul=captain_actual >= 10,
                optimal_id=int(optimal["player_id"]),
                optimal_name=str(optimal.get("player_name", "Unknown")),
                optimal_actual=optimal_actual,
                optimal_doubled=optimal_actual * 2,
                global_optimal_id=int(global_opt["player_id"]),
                global_optimal_name=str(global_opt.get("player_name", "Unknown")),
                global_optimal_actual=float(global_opt[self.target_column]),
                global_optimal_in_squad=global_opt_in_squad,
                regret=(optimal_actual - captain_actual) * 2,
                was_optimal=int(captain["player_id"]) == int(optimal["player_id"]),
            )
            results.append(result)
        
        summary = self._summarize(results)
        
        if verbose:
            print(summary.summary())
        
        return summary
    
    def _train_model(self, train_df: pd.DataFrame):
        """Train single LightGBM model for walk-forward backtesting.
        
        Returns:
            Trained LightGBM model
        """
        X = train_df[FEATURE_COLUMNS].values
        y = train_df[self.target_column].values
        
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=100,
        )
        return model
    
    def _predict(self, model, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return model.predict(X)
    
    def _summarize(
        self,
        results: List[CaptainGWResult],
    ) -> CaptainBacktestSummary:
        """Summarize backtest results."""
        if not results:
            return CaptainBacktestSummary(
                n_gameweeks=0,
                total_doubled_points=0,
                avg_doubled_points=0,
                optimal_doubled_points=0,
                hit_count=0,
                hit_rate=0,
                haul_count=0,
                haul_rate=0,
                squad_recall_count=0,
                squad_recall_rate=0,
                total_regret=0,
                avg_regret=0,
                gw_results=[],
            )
        
        n = len(results)
        total_doubled = sum(r.captain_doubled for r in results)
        optimal_doubled = sum(r.optimal_doubled for r in results)
        hit_count = sum(1 for r in results if r.was_optimal)
        haul_count = sum(1 for r in results if r.was_haul)
        squad_recall_count = sum(1 for r in results if r.global_optimal_in_squad)
        total_regret = sum(r.regret for r in results)
        
        return CaptainBacktestSummary(
            n_gameweeks=n,
            total_doubled_points=total_doubled,
            avg_doubled_points=total_doubled / n,
            optimal_doubled_points=optimal_doubled,
            hit_count=hit_count,
            hit_rate=hit_count / n,
            haul_count=haul_count,
            haul_rate=haul_count / n,
            squad_recall_count=squad_recall_count,
            squad_recall_rate=squad_recall_count / n,
            total_regret=total_regret,
            avg_regret=total_regret / n,
            gw_results=results,
        )
