"""Training + evaluation pipeline runner.

Orchestrates the complete training workflow:
1. Train models on training data
2. Evaluate on all datasets
3. Save results

Usage:
    python scripts/train_and_eval.py \\
        --datasets-dir storage/datasets \\
        --out-dir models/lightgbm_v1 \\
        --max-rounds 200
"""

import argparse
from pathlib import Path

from dugout.production.pipeline.trainer import Trainer as TrainingPipeline
from dugout.production.pipeline.evaluator import Evaluator


def main():
    """Run full training + evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate FPL prediction model"
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="storage/datasets",
        help="Directory with train/val/test CSVs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models/lightgbm_v1",
        help="Output directory for models",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=200,
        help="LightGBM boosting rounds",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of training data to use (for quick runs)",
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
    
    # Train
    print("\n[1/2] TRAINING")
    print("-" * 70)
    trainer = TrainingPipeline(
        datasets_dir=args.datasets_dir,
        out_dir=args.out_dir,
        max_rounds=args.max_rounds,
        sample_frac=args.sample_frac,
    )
    models = trainer.train()
    
    # Evaluate
    if not args.skip_eval:
        print("\n[2/2] EVALUATION")
        print("-" * 70)
        evaluator = Evaluator(
            model_dir=args.out_dir,
            datasets_dir=args.datasets_dir,
        )
        results = evaluator.evaluate_all()
        
        # Save results
        eval_path = evaluator.save_results(results)
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Models:     {args.out_dir}")
        print(f"Results:    {eval_path}")
    else:
        print("\n[2/2] EVALUATION SKIPPED")
        print("=" * 70)
        print(f"Models saved to {args.out_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
