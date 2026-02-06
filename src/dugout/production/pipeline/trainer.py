"""Model trainer for FPL ML.

ALL TRAINING LOGIC MUST LIVE HERE.
This is the single source of truth for production model training.

Decision-specific training:
    - train_captain_model(): Position-conditional features
    - train_transfer_model(): Baseline features
    - train_free_hit_model(): Baseline + cost

Key Classes:
    Trainer - Canonical trainer for all production models

Usage:
    from dugout.production.pipeline import Trainer
    
    trainer = Trainer()
    captain_model = trainer.train_captain_model(train_df)
    transfer_model = trainer.train_transfer_model(train_df)
    free_hit_model = trainer.train_free_hit_model(train_df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from dugout.production.features.views import (
    CAPTAIN_FEATURES,
    TRANSFER_FEATURES,
    FREE_HIT_FEATURES,
    DEFENSIVE_POSITIONS,
)
from dugout.production.models.captain_model import CaptainModel
from dugout.production.models.transfer_model import TransferModel
from dugout.production.models.free_hit_model import FreeHitModel


# =============================================================================
# Hyperparameters (centralized)
# =============================================================================

SINGLE_MODEL_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.07,
    "num_leaves": 8,
    "min_data_in_leaf": 150,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
}

P_PLAY_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.07,
    "num_leaves": 8,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
}

MU_POINTS_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.07,
    "num_leaves": 8,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
}


class Trainer:
    """Canonical trainer for all production models.
    
    ALL training flows through this class. No other file may fit models.
    
    Decision-specific training:
        - train_captain_model(): Position-conditional features
        - train_transfer_model(): Baseline features  
        - train_free_hit_model(): Baseline + cost
    """
    
    def __init__(self, max_rounds: int = 100):
        self.max_rounds = max_rounds
    
    # =========================================================================
    # Decision-Specific Training (Recommended)
    # =========================================================================
    
    def train_captain_model(self, train_df: pd.DataFrame) -> CaptainModel:
        """Train captain-specific model with position-conditional features.
        
        Uses CAPTAIN_FEATURES with defensive features (xgc_per90, clean_sheet_rate).
        Defensive features are zeroed for MID/FWD during inference (not training).
        
        Args:
            train_df: Training DataFrame with CAPTAIN_FEATURES and total_points.
            
        Returns:
            Trained CaptainModel instance.
        """
        feature_cols = CAPTAIN_FEATURES
        
        # Validate required columns
        required = ["total_points"] + feature_cols
        missing = [c for c in required if c not in train_df.columns]
        if missing:
            raise ValueError(f"Missing required columns for captain model: {missing}")
        
        X = train_df[feature_cols].fillna(0).values
        y = train_df["total_points"].values
        
        print(f"Training Captain model ({len(feature_cols)} features)...")
        lgb_train = lgb.Dataset(X, label=y)
        gbm = lgb.train(SINGLE_MODEL_PARAMS, lgb_train, num_boost_round=self.max_rounds)
        
        print(f"  → Trained on {len(y):,} rows")
        return CaptainModel(model_artifact=gbm, feature_cols=feature_cols)
    
    def train_transfer_model(self, train_df: pd.DataFrame) -> TransferModel:
        """Train transfer-specific model with baseline features.
        
        Uses TRANSFER_FEATURES (baseline, no cost, no defensive).
        
        Args:
            train_df: Training DataFrame with TRANSFER_FEATURES and total_points.
            
        Returns:
            Trained TransferModel instance.
        """
        feature_cols = TRANSFER_FEATURES
        
        # Validate required columns
        required = ["total_points"] + feature_cols
        missing = [c for c in required if c not in train_df.columns]
        if missing:
            raise ValueError(f"Missing required columns for transfer model: {missing}")
        
        X = train_df[feature_cols].fillna(0).values
        y = train_df["total_points"].values
        
        print(f"Training Transfer model ({len(feature_cols)} features)...")
        lgb_train = lgb.Dataset(X, label=y)
        gbm = lgb.train(SINGLE_MODEL_PARAMS, lgb_train, num_boost_round=self.max_rounds)
        
        print(f"  → Trained on {len(y):,} rows")
        return TransferModel(model_artifact=gbm, feature_cols=feature_cols)
    
    def train_free_hit_model(self, train_df: pd.DataFrame) -> FreeHitModel:
        """Train Free Hit-specific model with cost-aware features.
        
        Uses FREE_HIT_FEATURES (baseline + now_cost).
        
        Args:
            train_df: Training DataFrame with FREE_HIT_FEATURES and total_points.
            
        Returns:
            Trained FreeHitModel instance.
        """
        feature_cols = FREE_HIT_FEATURES
        
        # Validate required columns
        required = ["total_points"] + feature_cols
        missing = [c for c in required if c not in train_df.columns]
        if missing:
            raise ValueError(f"Missing required columns for free hit model: {missing}")
        
        X = train_df[feature_cols].fillna(0).values
        y = train_df["total_points"].values
        
        print(f"Training Free Hit model ({len(feature_cols)} features)...")
        lgb_train = lgb.Dataset(X, label=y)
        gbm = lgb.train(SINGLE_MODEL_PARAMS, lgb_train, num_boost_round=self.max_rounds)
        
        print(f"  → Trained on {len(y):,} rows")
        return FreeHitModel(model_artifact=gbm, feature_cols=feature_cols)
    
    def train_all_models(
        self,
        train_df: pd.DataFrame,
        model_dir: Optional[Path] = None,
    ) -> dict:
        """Train and save all decision-specific models.
        
        Args:
            train_df: Training DataFrame with all required features.
            model_dir: Directory to save models (default: production model dir).
            
        Returns:
            Dict mapping decision name to saved model path.
        """
        from dugout.production.config import MODEL_DIR
        
        if model_dir is None:
            model_dir = MODEL_DIR / "lightgbm_v2"
        
        print("=" * 60)
        print("Training all decision-specific models")
        print("=" * 60)
        
        paths = {}
        
        # Captain
        captain = self.train_captain_model(train_df)
        paths["captain"] = captain.save(model_dir)
        
        # Transfer
        transfer = self.train_transfer_model(train_df)
        paths["transfer"] = transfer.save(model_dir)
        
        # Free Hit
        free_hit = self.train_free_hit_model(train_df)
        paths["free_hit"] = free_hit.save(model_dir)
        
        print("=" * 60)
        print("All models saved:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        
        return paths
    
    # =========================================================================
    # Legacy Training (Deprecated - kept for backward compatibility)
    # =========================================================================
    
    def train(self, train_df: pd.DataFrame) -> lgb.Booster:
        """Train single LightGBM regressor (legacy mode).
        
        DEPRECATED: Use train_transfer_model() or train_captain_model() instead.
        
        Args:
            train_df: Training DataFrame with feature columns and total_points.
        
        Returns:
            Trained LightGBM Booster.
        """
        X = train_df[TRANSFER_FEATURES].fillna(0).values
        y = train_df["total_points"].values
        
        print("Training LightGBM (legacy single model)...")
        lgb_train = lgb.Dataset(X, label=y)
        
        return lgb.train(SINGLE_MODEL_PARAMS, lgb_train, num_boost_round=self.max_rounds)
    
    def train_residuals(
        self,
        train_df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> RandomForestRegressor:
        """Train residual model for uncertainty estimation.
        
        Args:
            train_df: Training DataFrame with total_points.
            predictions: Model predictions on training data.
        
        Returns:
            Trained RandomForest for residual prediction.
        """
        X = train_df[TRANSFER_FEATURES].fillna(0).values
        y = train_df["total_points"].values
        
        print("Training residual model...")
        residuals = np.abs(predictions - y)
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X, residuals)
        return rf


# =============================================================================
# Module-level convenience functions
# =============================================================================

def train_captain_model(train_df: pd.DataFrame) -> CaptainModel:
    """Train captain-specific model. See Trainer.train_captain_model."""
    return Trainer().train_captain_model(train_df)


def train_transfer_model(train_df: pd.DataFrame) -> TransferModel:
    """Train transfer-specific model. See Trainer.train_transfer_model."""
    return Trainer().train_transfer_model(train_df)


def train_free_hit_model(train_df: pd.DataFrame) -> FreeHitModel:
    """Train Free Hit-specific model. See Trainer.train_free_hit_model."""
    return Trainer().train_free_hit_model(train_df)


def train_all_models(train_df: pd.DataFrame, model_dir: Optional[Path] = None) -> dict:
    """Train and save all decision-specific models. See Trainer.train_all_models."""
    return Trainer().train_all_models(train_df, model_dir)


# Re-export for convenience
__all__ = [
    "Trainer",
    "CaptainModel",
    "TransferModel", 
    "FreeHitModel",
    "train_captain_model",
    "train_transfer_model",
    "train_free_hit_model",
    "train_all_models",
]
