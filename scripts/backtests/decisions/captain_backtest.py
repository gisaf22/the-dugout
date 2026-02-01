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
    PYTHONPATH=src python scripts/backtests/decisions/captain_backtest.py --compare-models
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from dugout.production.data.reader import DataReader
from dugout.production.features.builder import FeatureBuilder
from dugout.production.models.backtest import CaptainBacktester
from dugout.production.config import DEFAULT_DB_PATH


def run_backtest(df, model_type="legacy"):
    """Run captain backtest for a single model type."""
    print(f"\n{'='*70}")
    print(f"CAPTAIN BACKTEST: {model_type.upper()}")
    print(f"Decision Rule: argmax(predicted_points)")
    print(f"{'='*70}")
    
    backtester = CaptainBacktester(
        min_train_gws=5,
        target_column="total_points",
        model_type=model_type,
    )
    
    result = backtester.run(df, squad_size=15)
    return result


def compare_models(df):
    """Compare legacy vs two-stage model regret."""
    print("\n" + "=" * 80)
    print("LEGACY vs TWO-STAGE MODEL COMPARISON")
    print("Decision Rule: argmax(predicted_points)")
    print("=" * 80)
    
    # Run both model types
    legacy_bt = CaptainBacktester(
        min_train_gws=5,
        target_column="total_points",
        model_type="legacy",
    )
    two_stage_bt = CaptainBacktester(
        min_train_gws=5,
        target_column="total_points",
        model_type="two_stage",
    )
    
    print("\n--- Legacy Model ---")
    legacy = legacy_bt.run(df, squad_size=15, verbose=False)
    
    print("\n--- Two-Stage Model ---")
    two_stage = two_stage_bt.run(df, squad_size=15, verbose=False)
    
    # Per-GW comparison
    print("\n" + "=" * 90)
    print("PER-GW COMPARISON")
    print("=" * 90)
    print(f"{'GW':<5} {'Legacy Capt':<18} {'Pts':<6} {'Two-Stage Capt':<18} {'Pts':<6} {'Winner':<10}")
    print("-" * 90)
    
    legacy_wins, two_stage_wins, ties = 0, 0, 0
    
    for leg, ts in zip(legacy.gw_results, two_stage.gw_results):
        leg_pts = leg.captain_actual
        ts_pts = ts.captain_actual
        
        if leg_pts > ts_pts:
            winner = "Legacy"
            legacy_wins += 1
        elif ts_pts > leg_pts:
            winner = "Two-Stage"
            two_stage_wins += 1
        else:
            winner = "Tie"
            ties += 1
        
        print(
            f"{leg.gw:<5} "
            f"{leg.captain_name:<18} "
            f"{leg_pts:<6.0f} "
            f"{ts.captain_name:<18} "
            f"{ts_pts:<6.0f} "
            f"{winner:<10}"
        )
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("REGRET COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Legacy':<15} {'Two-Stage':<15} {'Winner':<10}")
    print("-" * 70)
    
    # Hit rate
    leg_hit = f"{legacy.hit_rate:.1%}"
    ts_hit = f"{two_stage.hit_rate:.1%}"
    hit_winner = "Legacy" if legacy.hit_rate > two_stage.hit_rate else ("Two-Stage" if two_stage.hit_rate > legacy.hit_rate else "Tie")
    print(f"{'Hit Rate':<25} {leg_hit:<15} {ts_hit:<15} {hit_winner:<10}")
    
    # Avg points
    leg_pts = f"{legacy.avg_doubled_points:.2f}"
    ts_pts = f"{two_stage.avg_doubled_points:.2f}"
    pts_winner = "Legacy" if legacy.avg_doubled_points > two_stage.avg_doubled_points else ("Two-Stage" if two_stage.avg_doubled_points > legacy.avg_doubled_points else "Tie")
    print(f"{'Avg Points (doubled)':<25} {leg_pts:<15} {ts_pts:<15} {pts_winner:<10}")
    
    # Avg regret (lower is better)
    leg_reg = f"{legacy.avg_regret:.2f}"
    ts_reg = f"{two_stage.avg_regret:.2f}"
    reg_winner = "Legacy" if legacy.avg_regret < two_stage.avg_regret else ("Two-Stage" if two_stage.avg_regret < legacy.avg_regret else "Tie")
    print(f"{'Avg Regret (↓ better)':<25} {leg_reg:<15} {ts_reg:<15} {reg_winner:<10}")
    
    # Total regret
    leg_total = f"{legacy.total_regret:.0f}"
    ts_total = f"{two_stage.total_regret:.0f}"
    total_winner = "Legacy" if legacy.total_regret < two_stage.total_regret else ("Two-Stage" if two_stage.total_regret < legacy.total_regret else "Tie")
    print(f"{'Total Regret (↓ better)':<25} {leg_total:<15} {ts_total:<15} {total_winner:<10}")
    
    # GW wins
    print(f"\n{'GW Wins:':<25} {legacy_wins:<15} {two_stage_wins:<15} {'Ties: ' + str(ties):<10}")
    
    print("\n" + "=" * 70)
    regret_diff = legacy.avg_regret - two_stage.avg_regret
    if regret_diff > 0:
        print(f"Two-Stage reduces avg regret by {regret_diff:.2f} pts/GW")
    elif regret_diff < 0:
        print(f"Legacy has lower avg regret by {-regret_diff:.2f} pts/GW")
    else:
        print("Both models have equal regret")
    print("=" * 70)
    
    return {"legacy": legacy, "two_stage": two_stage}


def main():
    parser = argparse.ArgumentParser(description="Captain Backtest")
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare legacy vs two-stage model regret"
    )
    parser.add_argument(
        "--model-type",
        choices=["legacy", "two_stage"],
        default="legacy",
        help="Model type to use (default: legacy)"
    )
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
    
    if args.compare_models:
        # Run model comparison
        return compare_models(df)
    else:
        # Run single model type
        result = run_backtest(df, args.model_type)
        
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
