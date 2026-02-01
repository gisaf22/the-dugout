"""FPL ML Pipeline.

End-to-end pipeline that orchestrates:
1. Data gathering (DataReader)
2. Feature building (FeatureBuilder)
3. Train/val/test split (DatasetBuilder)
4. Model training (TrainingPipeline)
5. Evaluation on all splits
6. Artifact saving

Usage:
    from dugout.production import Pipeline
    
    # Full pipeline
    Pipeline.run()
    
    # Or step by step
    pipeline = Pipeline()
    pipeline.gather_data()
    pipeline.build_features()
    pipeline.split()
    pipeline.train()
    pipeline.evaluate()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dugout.production.config import DEFAULT_DB_PATH, MODEL_DIR, STORAGE_DIR
from dugout.production.data import DataReader
from dugout.production.features import FeatureBuilder
from dugout.production.features.definitions import FEATURE_COLUMNS
from dugout.production.models import DatasetBuilder
from dugout.production.models.two_stage import (
    TwoStageModels, 
    save_two_stage_models, load_two_stage_models
)
from dugout.production.pipeline.trainer import Trainer


class Pipeline:
    """End-to-end FPL ML pipeline."""
    
    DATASETS_DIR = STORAGE_DIR / "datasets"
    MODEL_PATH = MODEL_DIR / "lightgbm_v1"
    
    def __init__(self, two_stage: bool = False, conditional_on_play: bool = False):
        """Initialize pipeline.
        
        Args:
            two_stage: If True, use epistemically-aligned two-stage modeling:
                p_play × mu_points. Research-validated approach that separates
                participation from performance. Default False for backward compatibility.
            conditional_on_play: (Legacy) If True, train only on rows where player appeared
                (minutes > 0). Ignored if two_stage=True.
        """
        self.db_path = DEFAULT_DB_PATH
        self.two_stage = two_stage
        self.conditional_on_play = conditional_on_play
        
        # State
        self.raw_df: Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.model = None  # lgb.Booster for legacy, TwoStageModels for two_stage
        self.residual_model = None
        self.metrics: Dict = {}
        
        # Load model if exists
        if self.two_stage:
            two_stage_file = self.MODEL_PATH / "two_stage_model.joblib"
            if two_stage_file.exists():
                self.model = load_two_stage_models(self.MODEL_PATH)
        else:
            model_file = self.MODEL_PATH / "model.joblib"
            if model_file.exists():
                data = joblib.load(model_file)
            self.model = data["gbm"]
            self.residual_model = data.get("residual_rf")
    
    def gather_data(self) -> pd.DataFrame:
        """Step 1: Extract raw gameweek data from database."""
        reader = DataReader(self.db_path)
        print(f"Database: {reader.db_path}")
        
        self.raw_df = reader.get_all_gw_data()
        print(f"Gathered {len(self.raw_df):,} rows, {self.raw_df['player_id'].nunique()} players")
        return self.raw_df
    
    def build_features(self) -> pd.DataFrame:
        """Step 2: Build features per player-gameweek."""
        if self.raw_df is None:
            raise ValueError("Call gather_data() first")
        
        builder = FeatureBuilder()
        self.feature_df = builder.build_training_set(self.raw_df)
        print(f"Built {len(self.feature_df):,} feature rows")
        return self.feature_df
    
    def split(self, test_gws: int = 4, val_gws: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Step 3: Split into train/val/test by gameweek.
        
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        if self.feature_df is None:
            raise ValueError("Call build_features() first")
        
        splitter = DatasetBuilder(test_gws=test_gws, val_gws=val_gws, gw_column="gw")
        datasets = splitter.create_splits(self.feature_df)
        
        self.train_df = datasets.train
        self.val_df = datasets.val
        self.test_df = datasets.test
        
        return self.train_df, self.val_df, self.test_df
    
    def train(self) -> None:
        """Step 4: Train model(s) and save artifacts.
        
        If two_stage=True: Trains p_play and mu_points separately.
        Otherwise: Trains single LightGBM regressor.
        """
        if self.train_df is None:
            raise ValueError("Call split() first")
        
        trainer = Trainer()
        
        if self.two_stage:
            # Epistemically-aligned two-stage training
            # Research-validated: separates participation from performance
            print("\n🎯 TWO-STAGE TRAINING (Epistemically Aligned)")
            self.model = trainer.train_two_stage(self.train_df)
            
            # Save two-stage models
            self.MODEL_PATH.mkdir(parents=True, exist_ok=True)
            save_two_stage_models(self.model, self.MODEL_PATH)
            
            print(f"Two-stage model saved to {self.MODEL_PATH}")
        else:
            # Legacy single-model training
            # Apply conditional training filter if enabled
            if self.conditional_on_play:
                train_data = self.train_df[self.train_df["minutes"] > 0].copy()
                print(f"Conditional training: {len(train_data):,} / {len(self.train_df):,} rows (minutes > 0)")
            else:
                train_data = self.train_df
            
            # Train LightGBM
            self.model = trainer.train(train_data)
            
            # Predict on training data (don't use self.predict yet - residual_model not ready)
            X = train_data[FEATURE_COLUMNS].values
            predictions = self.model.predict(X)
            
            # Train residual model (on same filtered data)
            self.residual_model = trainer.train_residuals(train_data, predictions)
            
            # Save
            self.MODEL_PATH.mkdir(parents=True, exist_ok=True)
            model_path = self.MODEL_PATH / "model.joblib"
            joblib.dump({
                "gbm": self.model,
                "residual_rf": self.residual_model,
                "feature_cols": FEATURE_COLUMNS,
            }, model_path)
            print(f"Model saved to {model_path}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run model predictions on a DataFrame.
        
        Output is always: predicted_points
        
        For two-stage: predicted_points = p_play × mu_points
        For legacy: predicted_points = gbm.predict(X)
        
        Args:
            df: DataFrame with feature columns (train, val, or test).
            
        Returns:
            DataFrame with predicted_points (and optionally p_play, mu_points, uncertainty).
        """
        result = df.copy()
        X = df[FEATURE_COLUMNS].values
        
        if self.two_stage:
            # Load two-stage model if needed
            if self.model is None or not isinstance(self.model, TwoStageModels):
                self.model = load_two_stage_models(self.MODEL_PATH)
            
            # Two-stage prediction: p_play × mu_points
            components = self.model.predict_components(X)
            result["p_play"] = components["p_play"]
            result["mu_points"] = components["mu_points"]
            result["predicted_points"] = components["predicted_points"]
        else:
            # Legacy single-model prediction
            if self.model is None:
                data = joblib.load(self.MODEL_PATH / "model.joblib")
                self.model = data["gbm"]
                self.residual_model = data.get("residual_rf")
            
            result["predicted_points"] = self.model.predict(X)
            
            if self.residual_model is not None:
                result["uncertainty"] = self.residual_model.predict(X)
        
        return result
    
    def evaluate(self) -> Dict:
        """Step 5: Evaluate model on all splits."""
        if self.train_df is None:
            raise ValueError("Call split() first")
        if self.model is None:
            raise ValueError("Call train() first")
        
        def eval_split(df: pd.DataFrame, name: str) -> Dict:
            pred_df = self.predict(df)
            y = df["total_points"].values
            pred = pred_df["predicted_points"].values
            
            mae = mean_absolute_error(y, pred)
            rmse = np.sqrt(mean_squared_error(y, pred))
            
            print(f"{name:5s}: MAE={mae:.3f}, RMSE={rmse:.3f}")
            return {"mae": mae, "rmse": rmse, "n": len(df)}
        
        print("\nEvaluation:")
        self.metrics = {
            "train": eval_split(self.train_df, "Train"),
            "val": eval_split(self.val_df, "Val"),
            "test": eval_split(self.test_df, "Test"),
        }
        
        # Add model type to metrics
        self.metrics["model_type"] = "two_stage" if self.two_stage else "single"
        
        return self.metrics
    
    def save_artifacts(self) -> None:
        """Step 6: Save evaluation report."""
        self.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        report_path = self.MODEL_PATH / "report.json"
        
        with open(report_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Report saved to {report_path}")
    
    @classmethod
    def run(cls, two_stage: bool = False) -> "Pipeline":
        """Run full pipeline end-to-end.
        
        Args:
            two_stage: If True, use epistemically-aligned two-stage modeling.
        """
        pipeline = cls(two_stage=two_stage)
        pipeline.gather_data()
        pipeline.build_features()
        pipeline.split()
        pipeline.train()
        pipeline.predict(pipeline.test_df)
        pipeline.evaluate()
        pipeline.save_artifacts()
        print("\nPipeline complete!")
        return pipeline
