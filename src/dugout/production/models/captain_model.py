"""Captain-specific prediction model.

Predictor meaning: μ(points | plays), position-conditional
Uses defensive features (xgc_per90, clean_sheet_rate) for DEF/GKP only.

See docs/research/decision_specific_modeling.md for ablation evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from dugout.production.models.base import BaseModel
from dugout.production.features.views import CAPTAIN_FEATURES, DEFENSIVE_POSITIONS


# Model artifact filename
MODEL_FILENAME = "captain_model.joblib"


class CaptainModel(BaseModel):
    """Position-conditional model for captain selection.
    
    Applies defensive features (xgc_per90, clean_sheet_rate) only to
    DEF/GKP positions. These are zeroed for MID/FWD.
    """
    
    def __init__(self, model_artifact: Any, feature_cols: Optional[list] = None):
        """Initialize captain model.
        
        Args:
            model_artifact: Trained LightGBM booster
            feature_cols: Feature columns (defaults to CAPTAIN_FEATURES)
        """
        super().__init__(
            model_artifact=model_artifact,
            feature_cols=feature_cols or CAPTAIN_FEATURES,
        )
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expected points with position-conditional features.
        
        Defensive features are zeroed for MID/FWD (positions 3, 4).
        
        Args:
            df: DataFrame with feature columns and 'position' column
            
        Returns:
            Array of predicted_points values
        """
        # Apply position conditioning
        df_conditioned = self._apply_position_conditioning(df)
        
        # Extract features and predict
        X = self._prepare_features(df_conditioned)
        return self._model.predict(X)
    
    def _apply_position_conditioning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zero defensive features for non-defensive positions.
        
        Args:
            df: DataFrame with position column
            
        Returns:
            DataFrame with conditioned features
        """
        df = df.copy()
        
        # Check if position column exists
        if "position" not in df.columns:
            # Without position, cannot condition - return as-is
            return df
        
        # Zero defensive features for MID/FWD (positions 3, 4)
        non_defensive_mask = ~df["position"].isin(DEFENSIVE_POSITIONS)
        
        for col in ["xgc_per90", "clean_sheet_rate"]:
            if col in df.columns:
                df.loc[non_defensive_mask, col] = 0.0
        
        return df
    
    @classmethod
    def load(cls, model_dir: Path) -> "CaptainModel":
        """Load captain model from disk.
        
        Args:
            model_dir: Directory containing captain_model.joblib
            
        Returns:
            Loaded CaptainModel instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = model_dir / MODEL_FILENAME
        
        if not path.exists():
            raise FileNotFoundError(
                f"Captain model not found at {path}. "
                f"Train with: python scripts/ops/train_and_eval.py --decision captain"
            )
        
        data = joblib.load(path)
        
        # Handle different storage formats
        if "model" in data:
            model = data["model"]
            feature_cols = data.get("feature_cols", CAPTAIN_FEATURES)
        elif "gbm" in data:
            model = data["gbm"]
            feature_cols = data.get("feature_cols", CAPTAIN_FEATURES)
        else:
            raise ValueError(f"Unknown model format in {path}")
        
        return cls(model_artifact=model, feature_cols=feature_cols)
    
    def save(self, model_dir: Path) -> Path:
        """Save captain model to disk.
        
        Args:
            model_dir: Directory to save to
            
        Returns:
            Path to saved model
        """
        return super().save(model_dir, MODEL_FILENAME)


__all__ = ["CaptainModel", "MODEL_FILENAME"]
