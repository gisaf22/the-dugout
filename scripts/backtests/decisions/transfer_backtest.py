#!/usr/bin/env python3
"""Transfer-IN Backtest — Decision Backtest.

Evaluates transfer-in recommendation policy quality using regret analysis.
This is a DECISION backtest, not a model backtest.

Question answered: Did the decision rule recommend the best transfer-in?
Metrics: Regret (oracle - chosen), hit rate

Decision Rule (Frozen): argmax(predicted_points)
Oracle: Player with max actual_points among eligible candidates

Usage:
    PYTHONPATH=src python scripts/backtests/decisions/transfer_backtest.py
    PYTHONPATH=src python scripts/backtests/decisions/transfer_backtest.py --start-gw 6 --end-gw 22
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
from dugout.production.models.predict import predict_points


@dataclass
class TransferGWResult:
    """Result for a single gameweek transfer decision."""
    gw: int
    chosen_player_id: int
    chosen_name: str
    chosen_predicted: float
    chosen_actual: float
    oracle_player_id: int
    oracle_name: str
    oracle_actual: float
    regret: float
    was_optimal: bool


@dataclass
class TransferBacktestSummary:
    """Summary of transfer backtest across all gameweeks."""
    gw_results: List[TransferGWResult]
    n_gameweeks: int
    total_regret: float
    mean_regret: float
    median_regret: float
    hit_rate: float
    hit_count: int
    high_regret_count: int  # regret >= 10
    high_regret_rate: float


def run_transfer_backtest(
    start_gw: int,
    end_gw: int,
    reader: Optional[DataReader] = None,
) -> TransferBacktestSummary:
    """Run transfer-in backtest over a range of gameweeks.
    
    For each GW in [start_gw, end_gw]:
    1. Build features using data up to GW-1
    2. Predict expected points for GW
    3. Select transfer-in candidate (argmax predicted)
    4. Compare to oracle (argmax actual)
    5. Compute regret
    
    Args:
        start_gw: First gameweek to evaluate
        end_gw: Last gameweek to evaluate
        reader: Optional DataReader instance
    
    Returns:
        TransferBacktestSummary with per-GW results and aggregates
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
        
        if len(eligible_df) == 0:
            print(f"  Skipping GW{target_gw}: no eligible players")
            continue
        
        # Predict expected points using unified interface
        eligible_df["predicted_points"] = predict_points(eligible_df)
        
        # Decision: argmax(predicted_points)
        chosen_idx = eligible_df["predicted_points"].idxmax()
        chosen = eligible_df.loc[chosen_idx]
        
        # Get actual points for target_gw
        actual_df = raw_df[raw_df["gw"] == target_gw][
            ["player_id", "total_points"]
        ].rename(columns={"total_points": "actual_points"})
        
        # Merge actuals to eligible pool
        eval_df = eligible_df.merge(actual_df, on="player_id", how="left")
        eval_df["actual_points"] = eval_df["actual_points"].fillna(0)
        
        # Get chosen actual points
        chosen_actual = eval_df.loc[
            eval_df["player_id"] == chosen["player_id"], "actual_points"
        ].values[0]
        
        # Oracle: argmax(actual_points) among eligible
        oracle_idx = eval_df["actual_points"].idxmax()
        oracle = eval_df.loc[oracle_idx]
        
        # Compute regret
        regret = oracle["actual_points"] - chosen_actual
        was_optimal = chosen["player_id"] == oracle["player_id"]
        
        gw_results.append(TransferGWResult(
            gw=target_gw,
            chosen_player_id=int(chosen["player_id"]),
            chosen_name=chosen.get("player_name", "Unknown"),
            chosen_predicted=float(chosen["predicted_points"]),
            chosen_actual=float(chosen_actual),
            oracle_player_id=int(oracle["player_id"]),
            oracle_name=oracle.get("player_name", "Unknown"),
            oracle_actual=float(oracle["actual_points"]),
            regret=float(regret),
            was_optimal=was_optimal,
        ))
    
    # Compute summary stats
    n_gws = len(gw_results)
    if n_gws == 0:
        raise ValueError("No gameweeks evaluated")
    
    regrets = [r.regret for r in gw_results]
    hits = [r for r in gw_results if r.was_optimal]
    high_regret = [r for r in gw_results if r.regret >= 10]
    
    return TransferBacktestSummary(
        gw_results=gw_results,
        n_gameweeks=n_gws,
        total_regret=sum(regrets),
        mean_regret=sum(regrets) / n_gws,
        median_regret=sorted(regrets)[n_gws // 2],
        hit_rate=len(hits) / n_gws,
        hit_count=len(hits),
        high_regret_count=len(high_regret),
        high_regret_rate=len(high_regret) / n_gws,
    )


def main():
    parser = argparse.ArgumentParser(description="Run transfer-in decision backtest")
    parser.add_argument("--start-gw", type=int, default=6, help="First GW to evaluate (default: 6)")
    parser.add_argument("--end-gw", type=int, default=None, help="Last GW to evaluate (default: latest)")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("TRANSFER-IN BACKTEST")
    print("Decision Rule: argmax(predicted_points)")
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
    summary = run_transfer_backtest(args.start_gw, end_gw, reader)
    
    # Per-GW details
    print("\n" + "-" * 90)
    print(f"{'GW':<5} {'Chosen':<20} {'Pred':<8} {'Actual':<8} {'Oracle':<20} {'Orc Pts':<8} {'Regret':<8}")
    print("-" * 90)
    
    for r in summary.gw_results:
        marker = "✓" if r.was_optimal else ""
        print(
            f"{r.gw:<5} "
            f"{r.chosen_name:<20} "
            f"{r.chosen_predicted:<8.2f} "
            f"{r.chosen_actual:<8.0f} "
            f"{r.oracle_name:<20} "
            f"{r.oracle_actual:<8.0f} "
            f"{r.regret:<8.0f} {marker}"
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Gameweeks evaluated: {summary.n_gameweeks}")
    print(f"  Hit Rate:            {summary.hit_rate:.1%} ({summary.hit_count}/{summary.n_gameweeks})")
    print(f"  Total Regret:        {summary.total_regret:.0f} pts")
    print(f"  Mean Regret:         {summary.mean_regret:.2f} pts/GW")
    print(f"  Median Regret:       {summary.median_regret:.1f} pts")
    print(f"  High Regret (≥10):   {summary.high_regret_rate:.1%} ({summary.high_regret_count}/{summary.n_gameweeks})")
    
    # Save to CSV
    output_path = Path("storage/production/reports/evaluation_transfer_backtest.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for r in summary.gw_results:
        rows.append({
            "gw": r.gw,
            "chosen_player_id": r.chosen_player_id,
            "chosen_name": r.chosen_name,
            "chosen_predicted": r.chosen_predicted,
            "chosen_actual": r.chosen_actual,
            "oracle_player_id": r.oracle_player_id,
            "oracle_name": r.oracle_name,
            "oracle_actual": r.oracle_actual,
            "regret": r.regret,
            "was_optimal": r.was_optimal,
        })
    
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return summary


if __name__ == "__main__":
    main()
