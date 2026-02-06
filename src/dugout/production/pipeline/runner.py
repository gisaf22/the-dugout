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
from dugout.production.features.definitions import FEATURE_COLUMNS, BASE_FEATURES, FREE_HIT_FEATURES
from dugout.production.models import DatasetBuilder
from dugout.production.pipeline.trainer import Trainer


class Pipeline:
    """End-to-end FPL ML pipeline."""
    
    DATASETS_DIR = STORAGE_DIR / "datasets"
    MODEL_PATH = MODEL_DIR / "lightgbm_v2"
    
    def __init__(self):
        """Initialize pipeline."""
        self.db_path = DEFAULT_DB_PATH
        
        # State
        self.raw_df: Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.decision_models: Optional[Dict] = None
        self.metrics: Dict = {}
    
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
        """Step 4: Train decision-specific models and save artifacts.
        
        Trains three decision-specific models:
            - Captain model (18 features, position-conditional)
            - Transfer model (16 features, baseline)
            - Free Hit model (17 features, includes cost)
        """
        if self.train_df is None:
            raise ValueError("Call split() first")
        
        trainer = Trainer()
        
        # Train decision-specific models
        print("\n🎯 DECISION-SPECIFIC TRAINING")
        print("=" * 60)
        self.decision_models = trainer.train_all_models(self.train_df, self.MODEL_PATH)
    
    def predict(self, df: pd.DataFrame, decision: str = "transfer") -> pd.DataFrame:
        """Run model predictions on a DataFrame.
        
        Uses decision-specific models via registry.
        
        Args:
            df: DataFrame with feature columns (train, val, or test).
            decision: Which decision model to use ("captain", "transfer", "free_hit")
            
        Returns:
            DataFrame with predicted_points column.
        """
        from dugout.production.models.registry import get_model
        
        result = df.copy()
        model = get_model(decision)
        result["predicted_points"] = model.predict(df)
        
        return result
    
    def evaluate(self) -> Dict:
        """Step 5: Evaluate model on all splits."""
        if self.train_df is None:
            raise ValueError("Call split() first")
        if self.decision_models is None:
            raise ValueError("Call train() first")
        
        def eval_split(df: pd.DataFrame, name: str) -> Dict:
            pred_df = self.predict(df, decision="transfer")
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
        self.metrics["model_type"] = "decision_specific"
        
        return self.metrics
    
    def save_artifacts(self) -> None:
        """Step 6: Save evaluation report."""
        self.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        report_path = self.MODEL_PATH / "report.json"
        
        with open(report_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Report saved to {report_path}")
    
    @classmethod
    def run(cls) -> "Pipeline":
        """Run full pipeline end-to-end."""
        pipeline = cls()
        pipeline.gather_data()
        pipeline.build_features()
        pipeline.split()
        pipeline.train()
        pipeline.predict(pipeline.test_df)
        pipeline.evaluate()
        pipeline.save_artifacts()
        print("\nPipeline complete!")
        return pipeline
