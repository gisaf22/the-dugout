"""Free Hit-specific prediction model.

Predictor meaning: μ(points), cost-aware (budget correlation matters)
Uses baseline features + cost for budget-constrained LP optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from dugout.production.models.base import BaseModel
from dugout.production.features.views import FREE_HIT_FEATURES


# Model artifact filename
MODEL_FILENAME = "free_hit_model.joblib"


class FreeHitModel(BaseModel):
    """Cost-aware model for Free Hit optimization.
    
    Includes cost feature because Free Hit is a budget-constrained
    selection problem where price correlates with player ceiling.
    """
    
    def __init__(self, model_artifact: Any, feature_cols: Optional[list] = None):
        """Initialize Free Hit model.
        
        Args:
            model_artifact: Trained LightGBM booster
            feature_cols: Feature columns (defaults to FREE_HIT_FEATURES)
        """
        super().__init__(
            model_artifact=model_artifact,
            feature_cols=feature_cols or FREE_HIT_FEATURES,
        )
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expected points (cost-aware).
        
        Args:
            df: DataFrame with feature columns including now_cost
            
        Returns:
            Array of predicted_points values
        """
        X = self._prepare_features(df)
        return self._model.predict(X)
    
    @classmethod
    def load(cls, model_dir: Path) -> "FreeHitModel":
        """Load Free Hit model from disk.
        
        Args:
            model_dir: Directory containing free_hit_model.joblib
            
        Returns:
            Loaded FreeHitModel instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = model_dir / MODEL_FILENAME
        
        if not path.exists():
            raise FileNotFoundError(
                f"Free Hit model not found at {path}. "
                f"Train with: python scripts/ops/train_and_eval.py --decision free_hit"
            )
        
        data = joblib.load(path)
        
        # Handle different storage formats
        if "model" in data:
            model = data["model"]
            feature_cols = data.get("feature_cols", FREE_HIT_FEATURES)
        elif "gbm" in data:
            model = data["gbm"]
            feature_cols = data.get("feature_cols", FREE_HIT_FEATURES)
        else:
            raise ValueError(f"Unknown model format in {path}")
        
        return cls(model_artifact=model, feature_cols=feature_cols)
    
    def save(self, model_dir: Path) -> Path:
        """Save Free Hit model to disk.
        
        Args:
            model_dir: Directory to save to
            
        Returns:
            Path to saved model
        """
        return super().save(model_dir, MODEL_FILENAME)


__all__ = ["FreeHitModel", "MODEL_FILENAME"]
