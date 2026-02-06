#!/usr/bin/env python3
"""Free Hit Backtest — Decision Backtest.

Evaluates Free Hit squad optimization policy quality using regret analysis.
This is a DECISION backtest, not a model backtest.

Question answered: Did the optimizer select a good squad?
Metrics: Regret (oracle XI points - chosen XI points)

Decision Rule (Frozen): LP maximize Σ(predicted_points) with mode=basic
Oracle: LP maximize Σ(actual_points) with same constraints

NOTE: Only starting XI points are evaluated (bench is ignored).

Usage:
    PYTHONPATH=src python scripts/backtests/decisions/free_hit_backtest.py
    PYTHONPATH=src python scripts/backtests/decisions/free_hit_backtest.py --start-gw 6 --end-gw 22
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import pandas as pd

from dugout.production.data.reader import DataReader
from dugout.production.features.builder import FeatureBuilder
from dugout.production.features.definitions import FEATURE_COLUMNS
from dugout.production.config import DEFAULT_DB_PATH
from dugout.production.models.squad import FreeHitOptimizer
from dugout.production.models.registry import get_model


@dataclass
class FreeHitGWResult:
    """Result for a single gameweek Free Hit decision."""
    gw: int
    chosen_xi_points: float  # Sum of actual points for chosen starting XI
    chosen_xi_predicted: float  # Sum of predicted points for chosen starting XI
    oracle_xi_points: float  # Sum of actual points for oracle starting XI
    regret: float
    chosen_formation: str
    oracle_formation: str
    chosen_captain: str
    oracle_captain: str


@dataclass
class FreeHitBacktestSummary:
    """Summary of Free Hit backtest across all gameweeks."""
    gw_results: List[FreeHitGWResult]
    n_gameweeks: int
    total_regret: float
    mean_regret: float
    median_regret: float
    chosen_total_points: float
    oracle_total_points: float
    high_regret_count: int  # regret >= 10
    high_regret_rate: float


def _compute_xi_actual_points(xi_list: List[dict], actuals_df: pd.DataFrame) -> float:
    """Compute total actual points for a starting XI."""
    total = 0.0
    for player in xi_list:
        pid = player["player_id"]
        actual = actuals_df.loc[actuals_df["player_id"] == pid, "actual_points"]
        if len(actual) > 0:
            total += actual.values[0]
    return total


def run_free_hit_backtest(
    start_gw: int,
    end_gw: int,
    reader: Optional[DataReader] = None,
) -> FreeHitBacktestSummary:
    """Run Free Hit backtest over a range of gameweeks.
    
    For each GW in [start_gw, end_gw]:
    1. Build features using data up to GW-1
    2. Run optimizer with predicted points → chosen squad
    3. Run optimizer with actual points → oracle squad
    4. Compare starting XI actual points
    5. Compute regret
    
    Args:
        start_gw: First gameweek to evaluate
        end_gw: Last gameweek to evaluate
        reader: Optional DataReader instance
    
    Returns:
        FreeHitBacktestSummary with per-GW results and aggregates
    """
    if reader is None:
        reader = DataReader(DEFAULT_DB_PATH)
    
    # Load all data once
    raw_df = reader.get_all_gw_data()
    available_gws = sorted(raw_df["gw"].unique())
    
    # Validate GW range
    if start_gw < min(available_gws) + 1:
        raise ValueError(f"start_gw must be >= {min(available_gws) + 1} (need GW-1 for features)")
    if end_gw > max(available_gws):
        raise ValueError(f"end_gw must be <= {max(available_gws)} (latest available)")
    
    gw_results = []
    
    for target_gw in range(start_gw, end_gw + 1):
        history_gw = target_gw - 1
        
        if history_gw not in available_gws:
            print(f"  Skipping GW{target_gw}: no GW{history_gw} data")
            continue
        if target_gw not in available_gws:
            print(f"  Skipping GW{target_gw}: no actual data for evaluation")
            continue
        
        # Build features using only history up to GW-1
        history_df = raw_df[raw_df["gw"] <= history_gw].copy()
        
        # Get target GW fixtures for is_home_next
        fixtures = reader.get_fixtures(gw=target_gw)
        fixture_map = {}
        for f in fixtures:
            fixture_map[f["team_h"]] = True
            fixture_map[f["team_a"]] = False
        
        fb = FeatureBuilder()
        latest_df = fb.build_for_prediction(history_df, fixture_map)
        
        # Merge status from history_gw
        player_status = history_df[history_df["gw"] == history_gw][
            ["player_id", "status"]
        ].drop_duplicates()
        latest_df = latest_df.merge(player_status, on="player_id", how="left")
        
        # Filter unavailable (same as production)
        unavailable = ["n", "i", "s", "u"]
        eligible_df = latest_df[~latest_df["status"].isin(unavailable)].copy()
        
        if len(eligible_df) < 15:
            print(f"  Skipping GW{target_gw}: only {len(eligible_df)} eligible players")
            continue
        
        # Predict expected points using FreeHitModel (baseline + cost)
        # Falls back to legacy predict_points if free_hit_model.joblib not available
        try:
            free_hit_model = get_model("free_hit")
            eligible_df["predicted_points"] = free_hit_model.predict(eligible_df)
        except FileNotFoundError:
            from dugout.production.models.predict import predict_points
            eligible_df["predicted_points"] = predict_points(eligible_df, model_variant="free_hit")
        
        # Get actual points for target_gw
        actual_df = raw_df[raw_df["gw"] == target_gw][
            ["player_id", "total_points"]
        ].rename(columns={"total_points": "actual_points"})
        
        # Merge actuals
        eligible_df = eligible_df.merge(actual_df, on="player_id", how="left")
        eligible_df["actual_points"] = eligible_df["actual_points"].fillna(0)
        
        # Prepare optimizer columns
        opt_df = eligible_df.copy()
        opt_df["cost"] = opt_df["now_cost"]
        opt_df["name"] = opt_df["player_name"]
        opt_df["team"] = opt_df["team_name"]
        opt_df["element_type"] = opt_df["position"]
        
        # Run optimizer with PREDICTED points (chosen squad)
        try:
            chosen_optimizer = FreeHitOptimizer(
                predictions_df=opt_df,
                budget=100.0,
            )
            chosen_result = chosen_optimizer.optimize()
        except Exception as e:
            print(f"  Skipping GW{target_gw}: optimizer failed (chosen) - {e}")
            continue
        
        if chosen_result is None:
            print(f"  Skipping GW{target_gw}: optimizer returned None (chosen)")
            continue
        
        # For oracle: replace predicted_points with actual_points
        oracle_df = opt_df.copy()
        oracle_df["predicted_points"] = oracle_df["actual_points"]
        
        try:
            oracle_optimizer = FreeHitOptimizer(
                predictions_df=oracle_df,
                budget=100.0,
            )
            oracle_result = oracle_optimizer.optimize()
        except Exception as e:
            print(f"  Skipping GW{target_gw}: optimizer failed (oracle) - {e}")
            continue
        
        if oracle_result is None:
            print(f"  Skipping GW{target_gw}: optimizer returned None (oracle)")
            continue
        
        # Compute actual points for chosen starting XI
        chosen_xi_actual = _compute_xi_actual_points(chosen_result.starting_xi, eligible_df)
        chosen_xi_predicted = sum(p["ev"] for p in chosen_result.starting_xi)
        
        # Compute actual points for oracle starting XI
        oracle_xi_actual = _compute_xi_actual_points(oracle_result.starting_xi, eligible_df)
        
        # Regret = oracle - chosen (both using actual points)
        regret = oracle_xi_actual - chosen_xi_actual
        
        gw_results.append(FreeHitGWResult(
            gw=target_gw,
            chosen_xi_points=chosen_xi_actual,
            chosen_xi_predicted=chosen_xi_predicted,
            oracle_xi_points=oracle_xi_actual,
            regret=regret,
            chosen_formation=chosen_result.formation,
            oracle_formation=oracle_result.formation,
            chosen_captain=chosen_result.captain["name"],
            oracle_captain=oracle_result.captain["name"],
        ))
    
    # Compute summary stats
    n_gws = len(gw_results)
    if n_gws == 0:
        raise ValueError("No gameweeks evaluated")
    
    regrets = [r.regret for r in gw_results]
    high_regret = [r for r in gw_results if r.regret >= 10]
    
    return FreeHitBacktestSummary(
        gw_results=gw_results,
        n_gameweeks=n_gws,
        total_regret=sum(regrets),
        mean_regret=sum(regrets) / n_gws,
        median_regret=sorted(regrets)[n_gws // 2],
        chosen_total_points=sum(r.chosen_xi_points for r in gw_results),
        oracle_total_points=sum(r.oracle_xi_points for r in gw_results),
        high_regret_count=len(high_regret),
        high_regret_rate=len(high_regret) / n_gws,
    )


def main():
    parser = argparse.ArgumentParser(description="Run Free Hit decision backtest")
    parser.add_argument("--start-gw", type=int, default=6, help="First GW to evaluate (default: 6)")
    parser.add_argument("--end-gw", type=int, default=None, help="Last GW to evaluate (default: latest)")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("FREE HIT BACKTEST")
    print("Decision Rule: LP maximize Σ(predicted_points), mode=basic")
    print("Evaluation: Starting XI actual points only")
    print("=" * 70)
    
    # Load data to determine end_gw if not specified
    reader = DataReader(DEFAULT_DB_PATH)
    raw_df = reader.get_all_gw_data()
    available_gws = sorted(raw_df["gw"].unique())
    
    end_gw = args.end_gw if args.end_gw else max(available_gws)
    
    print(f"\nData available: GW {min(available_gws)}-{max(available_gws)}")
    print(f"Evaluating: GW {args.start_gw}-{end_gw}")
    
    # Run backtest
    print("\nRunning backtest...")
    summary = run_free_hit_backtest(args.start_gw, end_gw, reader)
    
    # Per-GW details
    print("\n" + "-" * 100)
    print(f"{'GW':<5} {'Formation':<10} {'Pred XI':<10} {'Actual XI':<10} {'Oracle XI':<10} {'Regret':<10} {'Captain':<15}")
    print("-" * 100)
    
    for r in summary.gw_results:
        print(
            f"{r.gw:<5} "
            f"{r.chosen_formation:<10} "
            f"{r.chosen_xi_predicted:<10.1f} "
            f"{r.chosen_xi_points:<10.0f} "
            f"{r.oracle_xi_points:<10.0f} "
            f"{r.regret:<10.0f} "
            f"{r.chosen_captain:<15}"
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Gameweeks evaluated:   {summary.n_gameweeks}")
    print(f"  Chosen Total Points:   {summary.chosen_total_points:.0f} pts")
    print(f"  Oracle Total Points:   {summary.oracle_total_points:.0f} pts")
    print(f"  Total Regret:          {summary.total_regret:.0f} pts")
    print(f"  Mean Regret:           {summary.mean_regret:.2f} pts/GW")
    print(f"  Median Regret:         {summary.median_regret:.1f} pts")
    print(f"  High Regret (≥10):     {summary.high_regret_rate:.1%} ({summary.high_regret_count}/{summary.n_gameweeks})")
    print(f"  Capture Rate:          {summary.chosen_total_points / summary.oracle_total_points:.1%}")
    
    # Save to CSV
    output_path = Path("storage/production/reports/evaluation_free_hit_backtest.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for r in summary.gw_results:
        rows.append({
            "gw": r.gw,
            "chosen_formation": r.chosen_formation,
            "chosen_xi_predicted": r.chosen_xi_predicted,
            "chosen_xi_points": r.chosen_xi_points,
            "oracle_xi_points": r.oracle_xi_points,
            "regret": r.regret,
            "chosen_captain": r.chosen_captain,
            "oracle_captain": r.oracle_captain,
            "oracle_formation": r.oracle_formation,
        })
    
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return summary


if __name__ == "__main__":
    main()
