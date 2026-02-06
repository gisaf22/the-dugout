"""Base model interface for decision-specific models.

Provides shared interfaces only. No prediction logic here.
Each decision model inherits and implements its own prediction.

GUARDRAIL: No unified model. No feature switches inside predict().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for decision-specific models.
    
    Each decision (Captain, Transfer, Free Hit) has its own model class
    that inherits from this base and implements predict().
    """
    
    def __init__(self, model_artifact: Any, feature_cols: List[str]):
        """Initialize with trained model and feature columns.
        
        Args:
            model_artifact: The trained model (LightGBM, etc.)
            feature_cols: List of feature column names this model expects
        """
        self._model = model_artifact
        self._feature_cols = feature_cols.copy()
    
    @property
    def feature_cols(self) -> List[str]:
        """Feature columns this model expects."""
        return self._feature_cols.copy()
    
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expected points for players.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Array of predicted_points values
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, model_dir: Path) -> "BaseModel":
        """Load model from disk.
        
        Args:
            model_dir: Directory containing model artifacts
            
        Returns:
            Loaded model instance
        """
        pass
    
    def save(self, model_dir: Path, filename: str) -> Path:
        """Save model to disk.
        
        Args:
            model_dir: Directory to save to
            filename: Name of the model file
            
        Returns:
            Path to saved model
        """
        import joblib
        
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / filename
        
        joblib.dump({
            "model": self._model,
            "feature_cols": self._feature_cols,
        }, path)
        
        return path
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and validate features from DataFrame.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Feature matrix (n_samples, n_features)
            
        Raises:
            ValueError: If required columns are missing
        """
        missing = [c for c in self._feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        
        return df[self._feature_cols].fillna(0).values


__all__ = ["BaseModel"]
