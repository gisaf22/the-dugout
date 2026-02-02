"""Training + evaluation pipeline runner.

Orchestrates the complete training workflow:
1. Gather data from database
2. Build features
3. Split into train/val/test
4. Train models
5. Evaluate on all datasets
6. Save results

Usage:
    python scripts/ops/train_and_eval.py --two-stage
"""

import argparse
from pathlib import Path

from dugout.production.pipeline.runner import Pipeline
from dugout.production.features.definitions import FEATURE_COLUMNS


def main():
    """Run full training + evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate FPL prediction model"
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        default=True,
        help="Use two-stage model (p_play × mu_points). Default: True",
    )
    parser.add_argument(
        "--test-gws",
        type=int,
        default=4,
        help="Number of gameweeks for test set",
    )
    parser.add_argument(
        "--val-gws",
        type=int,
        default=4,
        help="Number of gameweeks for validation set",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("DUGOUT TRAINING + EVALUATION PIPELINE")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = Pipeline(two_stage=args.two_stage)
    
    # Step 1: Gather data
    print("\n[1/5] GATHERING DATA")
    print("-" * 70)
    pipeline.gather_data()
    
    # Step 2: Build features
    print("\n[2/5] BUILDING FEATURES")
    print("-" * 70)
    pipeline.build_features()
    
    # Step 3: Split
    print("\n[3/5] SPLITTING DATA")
    print("-" * 70)
    pipeline.split(test_gws=args.test_gws, val_gws=args.val_gws)
    
    # Step 4: Train
    print("\n[4/5] TRAINING")
    print("-" * 70)
    pipeline.train()
    
    # Step 5: Evaluate
    if not args.skip_eval:
        print("\n[5/5] EVALUATION")
        print("-" * 70)
        metrics = pipeline.evaluate()
        
        # Save artifacts
        pipeline.save_artifacts()
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Model: {pipeline.MODEL_PATH}")
        print(f"Features: {len(FEATURE_COLUMNS)} columns")
        if 'test' in metrics:
            print(f"Test MAE: {metrics['test']['mae']:.3f}")
    else:
        print("\n[5/5] EVALUATION SKIPPED")
        print("=" * 70)
        print(f"Models saved to {pipeline.MODEL_PATH}")
    
    return 0


if __name__ == "__main__":
    exit(main())
