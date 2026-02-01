"""Model evaluation against test/validation sets.

Provides separate evaluation logic (decoupled from training).

Key Classes:
    Evaluator - Loads models, runs predictions, computes metrics

Usage:
    from dugout.production.pipeline.evaluator import Evaluator
    
    evaluator = Evaluator(
        model_dir="models/lightgbm_v1",
        datasets_dir="storage/datasets"
    )
    metrics = evaluator.evaluate_test()
    metrics.print_summary()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dugout.production.features import FeatureBuilder, FeatureConfig
from dugout.production.models.baseline import predict_baseline


class EvaluationMetrics:
    """Container for evaluation results."""
    
    def __init__(
        self,
        dataset_name: str,
        model_mae: float,
        model_rmse: float,
        baseline_mae: float,
        baseline_rmse: float,
        mae_improvement: float,
        rmse_improvement: float,
        n_samples: int,
    ):
        self.dataset_name = dataset_name
        self.model_mae = model_mae
        self.model_rmse = model_rmse
        self.baseline_mae = baseline_mae
        self.baseline_rmse = baseline_rmse
        self.mae_improvement = mae_improvement
        self.rmse_improvement = rmse_improvement
        self.n_samples = n_samples
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset": self.dataset_name,
            "model_mae": round(self.model_mae, 4),
            "model_rmse": round(self.model_rmse, 4),
            "baseline_mae": round(self.baseline_mae, 4),
            "baseline_rmse": round(self.baseline_rmse, 4),
            "mae_improvement_pct": round(self.mae_improvement * 100, 2),
            "rmse_improvement_pct": round(self.rmse_improvement * 100, 2),
            "n_samples": self.n_samples,
        }
    
    def print_summary(self):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print(f"Evaluation: {self.dataset_name.upper()}")
        print(f"{'='*60}")
        print(f"Model MAE:              {self.model_mae:.4f}")
        print(f"Baseline MAE:           {self.baseline_mae:.4f}")
        print(f"MAE Improvement:        {self.mae_improvement*100:+.2f}%")
        print(f"\nModel RMSE:             {self.model_rmse:.4f}")
        print(f"Baseline RMSE:          {self.baseline_rmse:.4f}")
        print(f"RMSE Improvement:       {self.rmse_improvement*100:+.2f}%")
        print(f"\nSamples:                {self.n_samples}")


class Evaluator:
    """Evaluate trained models on test/val datasets.
    
    Attributes:
        model_dir: Directory with trained models
        datasets_dir: Directory with train/val/test CSVs
    """
    
    def __init__(
        self,
        model_dir: str = "models/lightgbm_v1",
        datasets_dir: str = "storage/datasets",
    ):
        """Initialize evaluator.
        
        Args:
            model_dir: Directory containing model.joblib and residual_model.joblib
            datasets_dir: Directory containing train/val/test CSVs
        """
        self.model_dir = Path(model_dir)
        self.datasets_dir = Path(datasets_dir)
        self.feature_builder = FeatureBuilder(FeatureConfig())
        
        # Load models
        self.gbm = joblib.load(self.model_dir / "model.joblib")
        self.residual_model = joblib.load(self.model_dir / "residual_model.joblib")
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a dataset CSV.
        
        Args:
            dataset_name: "train", "val", or "test"
            
        Returns:
            DataFrame from CSV
        """
        csv_path = self.datasets_dir / f"{dataset_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}")
        return pd.read_csv(csv_path)
    
    def evaluate(self, dataset_name: str = "test") -> EvaluationMetrics:
        """Evaluate on a dataset.
        
        Args:
            dataset_name: "train", "val", or "test"
            
        Returns:
            EvaluationMetrics with performance summary
        """
        print(f"\nEvaluating on {dataset_name} set...")
        
        # Load and build features
        raw_df = self.load_dataset(dataset_name)
        feat_df, feature_cols = self.feature_builder.build_from_dataframe(raw_df)
        
        # Extract X, y
        X = feat_df[feature_cols].values
        y = feat_df["target_next_gw_points"].astype(float).values
        
        # Predict
        y_pred = self.gbm.predict(X, num_iteration=self.gbm.best_iteration)
        baseline_pred = predict_baseline(feat_df)
        
        # Compute metrics
        model_mae = mean_absolute_error(y, y_pred)
        model_rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        baseline_mae = mean_absolute_error(y, baseline_pred)
        baseline_rmse = float(np.sqrt(mean_squared_error(y, baseline_pred)))
        
        # Compute improvements (negative is bad)
        mae_improvement = (baseline_mae - model_mae) / baseline_mae
        rmse_improvement = (baseline_rmse - model_rmse) / baseline_rmse
        
        metrics = EvaluationMetrics(
            dataset_name=dataset_name,
            model_mae=float(model_mae),
            model_rmse=model_rmse,
            baseline_mae=float(baseline_mae),
            baseline_rmse=baseline_rmse,
            mae_improvement=mae_improvement,
            rmse_improvement=rmse_improvement,
            n_samples=len(y),
        )
        
        return metrics
    
    def evaluate_all(self) -> Dict[str, EvaluationMetrics]:
        """Evaluate on all datasets (train, val, test).
        
        Returns:
            Dictionary: {"train": metrics, "val": metrics, "test": metrics}
        """
        results = {}
        for dataset_name in ["train", "val", "test"]:
            try:
                metrics = self.evaluate(dataset_name)
                metrics.print_summary()
                results[dataset_name] = metrics
            except FileNotFoundError as e:
                print(f"  Skipping {dataset_name}: {e}")
        
        return results
    
    def save_results(
        self,
        results: Dict[str, EvaluationMetrics],
        out_path: Optional[str] = None,
    ) -> str:
        """Save evaluation results to JSON.
        
        Args:
            results: Dictionary of evaluation metrics
            out_path: Path to save (default: model_dir/evaluation.json)
            
        Returns:
            Path to saved file
        """
        if out_path is None:
            out_path = self.model_dir / "evaluation.json"
        else:
            out_path = Path(out_path)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = {name: m.to_dict() for name, m in results.items()}
        
        with open(out_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {out_path}")
        return str(out_path)
