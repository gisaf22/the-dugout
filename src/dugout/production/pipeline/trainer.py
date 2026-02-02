"""Model trainer for FPL ML.

ALL TRAINING LOGIC MUST LIVE HERE.
This is the single source of truth for production model training.

Supports two training modes:
    1. Single model (legacy): Train one regressor on all data
    2. Two-stage (epistemically aligned): Separate participation/performance

Two model variants:
    - Base model: For Captain/Transfer (no cost - pure ranking)
    - Free Hit model: For Free Hit optimization (includes cost)

Key Classes:
    Trainer - Canonical trainer for all production models

Usage:
    from dugout.production import Trainer
    
    trainer = Trainer()
    base_model = trainer.train_two_stage(train_df)  # no cost
    fh_model = trainer.train_two_stage(train_df, include_cost=True)  # with cost
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from dugout.production.features.definitions import BASE_FEATURES, FREE_HIT_FEATURES, FEATURE_COLUMNS
from dugout.production.models.two_stage import TwoStageModels


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
    """
    
    def __init__(self, max_rounds: int = 100):
        self.max_rounds = max_rounds
    
    def train(self, train_df: pd.DataFrame) -> lgb.Booster:
        """Train single LightGBM regressor (legacy mode).
        
        Args:
            train_df: Training DataFrame with feature columns and total_points.
        
        Returns:
            Trained LightGBM Booster.
        """
        X = train_df[FEATURE_COLUMNS].values
        y = train_df["total_points"].values
        
        print("Training LightGBM (single model)...")
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
        X = train_df[FEATURE_COLUMNS].values
        y = train_df["total_points"].values
        
        print("Training residual model...")
        residuals = np.abs(predictions - y)
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X, residuals)
        return rf
    
    def train_two_stage(
        self, 
        train_df: pd.DataFrame, 
        include_cost: bool = False
    ) -> TwoStageModels:
        """Train epistemically-aligned two-stage models.
        
        Research-validated approach:
            1. p_play: P(minutes > 0) on all rows
            2. mu_points: E[points | plays] on rows where minutes > 0
        
        Final prediction: p_play × mu_points
        
        Args:
            train_df: Training DataFrame with features, 'minutes', 'total_points'
            include_cost: If True, include now_cost (for Free Hit model)
            
        Returns:
            TwoStageModels container
        """
        # Select feature set based on model variant
        feature_cols = FREE_HIT_FEATURES if include_cost else BASE_FEATURES
        model_name = "Free Hit" if include_cost else "Base"
        
        # Validate required columns
        required = ["minutes", "total_points"] + feature_cols
        missing = [c for c in required if c not in train_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        X_all = train_df[feature_cols].values
        
        # 1. Train p_play: P(minutes > 0) on ALL rows
        print(f"Training p_play ({model_name} model)...")
        y_play = (train_df["minutes"] > 0).astype(int).values
        lgb_train_play = lgb.Dataset(X_all, label=y_play)
        
        p_play = lgb.train(
            P_PLAY_PARAMS,
            lgb_train_play,
            num_boost_round=self.max_rounds,
        )
        
        play_rate = y_play.mean()
        print(f"  → Trained on {len(y_play):,} rows, play_rate={play_rate:.2%}")
        
        # 2. Train mu_points: E[points | plays] on rows where minutes > 0
        print(f"Training mu_points ({model_name} model)...")
        play_mask = train_df["minutes"] > 0
        train_played = train_df[play_mask]
        
        X_played = train_played[feature_cols].values
        y_points = train_played["total_points"].values
        lgb_train_points = lgb.Dataset(X_played, label=y_points)
        
        mu_points = lgb.train(
            MU_POINTS_PARAMS,
            lgb_train_points,
            num_boost_round=self.max_rounds,
        )
        
        print(f"  → Trained on {len(y_points):,} / {len(train_df):,} rows (minutes > 0)")
        
        return TwoStageModels(
            p_play=p_play,
            mu_points=mu_points,
            feature_cols=feature_cols.copy(),
        )


# Re-export for convenience
__all__ = ["Trainer", "TwoStageModels"]
