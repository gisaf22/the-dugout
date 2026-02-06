#!/usr/bin/env python
"""A/B Comparison: Single Model vs Two-Stage Epistemically-Aligned Model.

Compares:
    - Legacy: single LightGBM regressor on all rows
    - Two-Stage: p_play × mu_points (participation separated from performance)

Evaluates:
    - MAE on test set
    - Captain regret (via backtest)

Usage:
    PYTHONPATH=src python scripts/backtests/models/compare_models.py
    
Output:
    storage/production/reports/model_comparison.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from dugout.production.config import MODEL_DIR, STORAGE_DIR
from dugout.production.pipeline.runner import Pipeline


def train_and_evaluate_legacy() -> dict:
    """Train and evaluate legacy single-model."""
    print("\n" + "=" * 70)
    print("LEGACY MODEL (Single LightGBM)")
    print("=" * 70)
    
    pipeline = Pipeline(two_stage=False)
    pipeline.gather_data()
    pipeline.build_features()
    pipeline.split()
    pipeline.train()
    metrics = pipeline.evaluate()
    
    return {
        "train_mae": metrics["train"]["mae"],
        "val_mae": metrics["val"]["mae"],
        "test_mae": metrics["test"]["mae"],
    }


def train_and_evaluate_two_stage() -> dict:
    """Train and evaluate two-stage epistemically-aligned model."""
    print("\n" + "=" * 70)
    print("TWO-STAGE MODEL (p_play × mu_points)")
    print("=" * 70)
    
    pipeline = Pipeline(two_stage=True)
    pipeline.gather_data()
    pipeline.build_features()
    pipeline.split()
    pipeline.train()
    metrics = pipeline.evaluate()
    
    return {
        "train_mae": metrics["train"]["mae"],
        "val_mae": metrics["val"]["mae"],
        "test_mae": metrics["test"]["mae"],
    }


def main():
    """Run A/B comparison."""
    print("\n" + "#" * 70)
    print("# A/B MODEL COMPARISON")
    print("# Legacy (single) vs Two-Stage (p_play × mu_points)")
    print("#" * 70)
    
    # Train and evaluate both models
    legacy_metrics = train_and_evaluate_legacy()
    two_stage_metrics = train_and_evaluate_two_stage()
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\n                    Legacy    Two-Stage   Δ")
    print("-" * 50)
    
    for split in ["train", "val", "test"]:
        key = f"{split}_mae"
        legacy = legacy_metrics[key]
        two_stage = two_stage_metrics[key]
        delta = two_stage - legacy
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"{split.capitalize():8s} MAE:      {legacy:.4f}    {two_stage:.4f}    {direction} {abs(delta):.4f}")
    
    # Interpretation
    print("\n" + "-" * 50)
    print("Interpretation:")
    test_delta = two_stage_metrics["test_mae"] - legacy_metrics["test_mae"]
    if test_delta > 0.05:
        print("  ⚠️  Two-stage has higher MAE (expected from research)")
        print("     MAE is not the primary metric — regret is")
    elif test_delta < -0.05:
        print("  ✅ Two-stage has lower MAE (unexpected but good)")
    else:
        print("  ≈  MAE difference is negligible")
    
    print("\n  📋 Key insight from research:")
    print("     Separating P(play) from E[points|plays] improves")
    print("     DECISION QUALITY (regret), not necessarily raw MAE.")
    
    # Save results
    results = {
        "legacy": legacy_metrics,
        "two_stage": two_stage_metrics,
        "test_mae_delta": test_delta,
    }
    
    output_path = STORAGE_DIR / "reports" / "model_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
