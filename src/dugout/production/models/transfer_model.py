"""Transfer-specific prediction model.

Predictor meaning: μ(points), unconditional baseline
Uses baseline features only, no cost, no defensive features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from dugout.production.models.base import BaseModel
from dugout.production.features.views import TRANSFER_FEATURES


# Model artifact filename
MODEL_FILENAME = "transfer_model.joblib"


class TransferModel(BaseModel):
    """Baseline model for transfer-in recommendations.
    
    Uses unconditional baseline features. No position conditioning.
    Pure ranking by expected points.
    """
    
    def __init__(self, model_artifact: Any, feature_cols: Optional[list] = None):
        """Initialize transfer model.
        
        Args:
            model_artifact: Trained LightGBM booster
            feature_cols: Feature columns (defaults to TRANSFER_FEATURES)
        """
        super().__init__(
            model_artifact=model_artifact,
            feature_cols=feature_cols or TRANSFER_FEATURES,
        )
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expected points (unconditional).
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Array of predicted_points values
        """
        X = self._prepare_features(df)
        return self._model.predict(X)
    
    @classmethod
    def load(cls, model_dir: Path) -> "TransferModel":
        """Load transfer model from disk.
        
        Args:
            model_dir: Directory containing transfer_model.joblib
            
        Returns:
            Loaded TransferModel instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = model_dir / MODEL_FILENAME
        
        if not path.exists():
            raise FileNotFoundError(
                f"Transfer model not found at {path}. "
                f"Train with: python scripts/ops/train_and_eval.py --decision transfer"
            )
        
        data = joblib.load(path)
        
        # Handle different storage formats
        if "model" in data:
            model = data["model"]
            feature_cols = data.get("feature_cols", TRANSFER_FEATURES)
        elif "gbm" in data:
            model = data["gbm"]
            feature_cols = data.get("feature_cols", TRANSFER_FEATURES)
        else:
            raise ValueError(f"Unknown model format in {path}")
        
        return cls(model_artifact=model, feature_cols=feature_cols)
    
    def save(self, model_dir: Path) -> Path:
        """Save transfer model to disk.
        
        Args:
            model_dir: Directory to save to
            
        Returns:
            Path to saved model
        """
        return super().save(model_dir, MODEL_FILENAME)


__all__ = ["TransferModel", "MODEL_FILENAME"]
