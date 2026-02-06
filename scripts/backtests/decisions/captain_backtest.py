#!/usr/bin/env python3
"""Captain Backtest — Decision Backtest.

Evaluates captain selection policy quality using regret analysis.
This is a DECISION backtest, not a model backtest.

Question answered: Did the decision rule pick the right captain?
Metrics: Regret (oracle - chosen), hit rate, haul capture rate

Frozen Decision Rule: Captain = argmax(predicted_points)

See README.md in this directory for conceptual documentation.

Usage:
    PYTHONPATH=src python scripts/backtests/decisions/captain_backtest.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from dugout.production.data.reader import DataReader
from dugout.production.features.builder import FeatureBuilder
from dugout.production.models.backtest import CaptainBacktester
from dugout.production.config import DEFAULT_DB_PATH


def run_backtest(df):
    """Run captain backtest."""
    print(f"\n{'='*70}")
    print("CAPTAIN BACKTEST")
    print(f"Decision Rule: argmax(predicted_points)")
    print(f"{'='*70}")
    
    backtester = CaptainBacktester(
        min_train_gws=5,
        target_column="total_points",
    )
    
    result = backtester.run(df, squad_size=15)
    return result


def main():
    parser = argparse.ArgumentParser(description="Captain Backtest")
    args = parser.parse_args()
    
    # Step 1: Load and build features
    print("Loading data...")
    reader = DataReader(DEFAULT_DB_PATH)
    raw_df = reader.get_all_gw_data()
    print(f"Loaded {len(raw_df):,} raw rows")
    
    print("Building features...")
    builder = FeatureBuilder()
    df = builder.build_training_set(raw_df)
    print(f"Built {len(df):,} feature rows")
    
    # Check GW range
    gws = sorted(df["gw"].unique())
    print(f"GW range: {gws[0]}-{gws[-1]} ({len(gws)} weeks)")
    
    # Run backtest
    result = run_backtest(df)
    
    # Show per-GW details
    print(f"\nPer-GW Details:")
    print("-" * 90)
    print(f"{'GW':<5} {'Captain':<20} {'Pred':<8} {'Actual':<8} {'Optimal':<20} {'Opt Pts':<8} {'Regret':<8}")
    print("-" * 90)
    
    for r in result.gw_results:
        marker = "✓" if r.was_optimal else ""
        print(
            f"{r.gw:<5} "
            f"{r.captain_name:<20} "
            f"{r.captain_predicted:<8.2f} "
            f"{r.captain_actual:<8.0f} "
            f"{r.optimal_name:<20} "
            f"{r.optimal_actual:<8.0f} "
            f"{r.regret:<8.0f} {marker}"
        )
    
    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Hit Rate:   {result.hit_rate:.1%} ({result.hit_count}/{result.n_gameweeks})")
    print(f"  Haul Rate:  {result.haul_rate:.1%} ({result.haul_count}/{result.n_gameweeks})")
    print(f"  Avg Points: {result.avg_doubled_points:.2f} (doubled)")
    print(f"  Avg Regret: {result.avg_regret:.2f}")
    
    return result


if __name__ == "__main__":
    main()
