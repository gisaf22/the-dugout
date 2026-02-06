"""Legacy prediction interface (DEPRECATED).

Use dugout.production.models.registry.get_model() instead.

This module is kept for backward compatibility with decision modules
that haven't yet migrated to the registry-based approach.

DEPRECATED: This file will be removed in a future version.
Use the decision-specific models via registry:
    from dugout.production.models.registry import get_model
    model = get_model("captain")  # or "transfer", "free_hit"
    predictions = model.predict(df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal
import warnings

import joblib
import numpy as np
import pandas as pd

from dugout.production.config import MODEL_DIR
from dugout.production.features.definitions import (
    FEATURE_COLUMNS, BASE_FEATURES, FREE_HIT_FEATURES,
    CAPTAIN_FEATURES, DEFENSIVE_POSITIONS
)

# Type alias for model selection
ModelVariant = Literal["base", "free_hit", "captain"]

# Module-level cache for last used model type (for logging)
_last_model_type: Optional[str] = None


def _apply_position_conditional(df: pd.DataFrame) -> pd.DataFrame:
    """Zero defensive features for MID/FWD positions."""
    df = df.copy()
    
    if 'position' not in df.columns:
        return df
    
    mask = ~df['position'].isin(DEFENSIVE_POSITIONS)
    for col in ['xgc_per90', 'clean_sheet_rate']:
        if col in df.columns:
            df.loc[mask, col] = 0.0
    
    return df


def get_active_model_type(model_dir: Optional[Path] = None) -> str:
    """Return which model type would be used for predictions.
    
    DEPRECATED: Use registry.get_model() instead.
    """
    warnings.warn(
        "get_active_model_type() is deprecated. Use registry.get_model() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return "decision_specific"


def predict_points(
    df: pd.DataFrame,
    model_dir: Optional[Path] = None,
    model_variant: ModelVariant = "base",
) -> np.ndarray:
    """Predict expected points for players.
    
    DEPRECATED: Use registry.get_model() instead:
        from dugout.production.models.registry import get_model
        model = get_model("captain")  # or "transfer", "free_hit"
        predictions = model.predict(df)
    
    Args:
        df: DataFrame with feature columns
        model_dir: Directory containing models
        model_variant: "base", "captain", or "free_hit"
        
    Returns:
        Array of predicted_points values
    """
    global _last_model_type
    
    if model_dir is None:
        model_dir = MODEL_DIR / "lightgbm_v2"
    
    # Map variant to model file and features
    if model_variant == "free_hit":
        model_path = model_dir / "free_hit_model.joblib"
        feature_cols = FREE_HIT_FEATURES
        _last_model_type = "free_hit_model"
    elif model_variant == "captain":
        model_path = model_dir / "captain_model.joblib"
        feature_cols = CAPTAIN_FEATURES
        _last_model_type = "captain_model"
        df = _apply_position_conditional(df)
    else:  # base/transfer
        model_path = model_dir / "transfer_model.joblib"
        if not model_path.exists():
            # Fallback to legacy model.joblib
            model_path = model_dir / "model.joblib"
            feature_cols = FEATURE_COLUMNS
            _last_model_type = "legacy"
        else:
            feature_cols = BASE_FEATURES
            _last_model_type = "transfer_model"
    
    # Prepare features
    X = df[feature_cols].fillna(0).values
    
    # Load and predict
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. "
            f"Run trainer.train_all_models() first."
        )
    
    data = joblib.load(model_path)
    
    # Handle different model formats
    if isinstance(data, dict):
        if "gbm" in data:
            return data["gbm"].predict(X)
        elif "model" in data:
            return data["model"].predict(X)
    
    # Direct booster
    return data.predict(X)


def get_last_model_type() -> Optional[str]:
    """Return the model type used in the last predict_points() call.
    
    DEPRECATED: Model type is now always decision-specific.
    """
    return _last_model_type
