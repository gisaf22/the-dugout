#!/usr/bin/env python3
"""Walk-Forward Validation — Model Backtest.

Evaluates predictive stability over time using expanding-window training.
This is a MODEL backtest, not a decision backtest.

Question answered: How well do predictions generalize to unseen gameweeks?
Metrics: MAE, RMSE, Spearman correlation, calibration

See README.md in this directory for conceptual documentation.

Usage:
    PYTHONPATH=src python scripts/backtests/models/walk_forward_validation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from dugout.production.data.reader import DataReader
from dugout.production.features.builder import FeatureBuilder
from dugout.production.models.walk_forward import BacktestRunner
from dugout.production.analysis.decisions.regret_analysis import RegretAnalyzer
from dugout.production.config import DEFAULT_DB_PATH


def main():
    # Step 1: Load and build features
    print("Loading data...")
    reader = DataReader(DEFAULT_DB_PATH)
    raw_df = reader.get_all_gw_data()
    print(f"Loaded {len(raw_df):,} raw rows")
    
    print("Building features...")
    builder = FeatureBuilder()
    df = builder.build_training_set(raw_df)
    print(f"Built {len(df):,} feature rows")
    
    # Step 2: Run walk-forward backtest (with calibration)
    print("\nRunning walk-forward backtest...")
    runner = BacktestRunner(min_train_gws=5, target_column="total_points")
    summary = runner.run(df, store_predictions=True, calibrate=True)
    
    # Step 3: Print summary
    summary.print_summary()
    
    # Step 4: Regret analysis
    print("\n" + "=" * 60)
    print("REGRET ANALYSIS")
    print("=" * 60)
    
    analyzer = RegretAnalyzer()
    report = analyzer.analyze(summary)
    
    print(report.summary())
    
    # Step 5: Bucket A deep dive
    bucket_a_profile = analyzer.profile_bucket_a(summary)
    print(bucket_a_profile.summary())
    
    return summary, report, bucket_a_profile


if __name__ == "__main__":
    main()
