"""Prediction explainer for FPL model.

Provides interpretability tools to understand why the model made
specific predictions. Helps build trust and identify issues.

Key Classes:
    PredictionExplainer - Main explainer class
    FeatureContribution - Single feature's impact on prediction
    PredictionBreakdown - Full breakdown of a prediction

Capabilities:
    - Feature importance (global model importance)
    - Local explanations (why this player got this score)
    - Contribution plots (visualize feature impacts)

Usage:
    from dugout.production.analysis.models import PredictionExplainer
    
    explainer = PredictionExplainer(model)
    breakdown = explainer.explain(player_features)
    print(breakdown.top_contributors(n=5))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FeatureContribution:
    """Contribution of a single feature to a prediction."""
    
    feature_name: str
    feature_value: float
    contribution: float  # SHAP-style contribution to prediction
    
    def __repr__(self) -> str:
        sign = "+" if self.contribution >= 0 else ""
        return f"{self.feature_name}: {self.feature_value:.2f} ({sign}{self.contribution:.2f})"


@dataclass 
class PredictionBreakdown:
    """Full breakdown of a prediction."""
    
    player_id: int
    player_name: str
    predicted_points: float
    base_value: float  # Intercept / baseline
    contributions: List[FeatureContribution]
    top_positive: List[str]  # Top features pushing prediction up
    top_negative: List[str]  # Top features pushing prediction down
    
    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "predicted_points": round(self.predicted_points, 2),
            "base_value": round(self.base_value, 2),
            "top_positive_factors": self.top_positive,
            "top_negative_factors": self.top_negative,
            "contributions": [
                {"feature": c.feature_name, "value": c.feature_value, "contribution": c.contribution}
                for c in self.contributions[:10]  # Top 10
            ]
        }


class PredictionExplainer:
    """Explain model predictions using feature importance.
    
    Currently uses LightGBM's built-in feature importance.
    Can be extended to use SHAP for more detailed explanations.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """Initialize with trained model.
        
        Args:
            model: Trained LightGBM model
            feature_names: List of feature column names
        """
        self.model = model
        self.feature_names = feature_names
        self._importance_cache: Optional[Dict[str, float]] = None
    
    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """Get global feature importance.
        
        Args:
            importance_type: "gain" (default) or "split"
            
        Returns:
            Dict mapping feature names to importance scores (normalized)
        """
        if self._importance_cache is not None:
            return self._importance_cache
        
        try:
            importance = self.model.feature_importance(importance_type=importance_type)
            total = sum(importance)
            if total > 0:
                normalized = importance / total
            else:
                normalized = importance
            
            self._importance_cache = dict(zip(self.feature_names, normalized))
            return self._importance_cache
        except Exception:
            # Fallback for models without feature_importance
            return {name: 1.0 / len(self.feature_names) for name in self.feature_names}
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features.
        
        Returns:
            List of (feature_name, importance) tuples
        """
        importance = self.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]
    
    def explain_prediction(
        self,
        features: pd.Series,
        player_name: str = "Unknown",
        player_id: int = 0,
    ) -> PredictionBreakdown:
        """Explain a single prediction.
        
        Uses feature importance as proxy for contribution.
        For more accurate explanations, consider SHAP integration.
        
        Args:
            features: Feature values for the player
            player_name: Player name for display
            player_id: Player ID
            
        Returns:
            PredictionBreakdown with feature contributions
        """
        importance = self.get_feature_importance()
        
        # Get prediction
        X = features[self.feature_names].values.reshape(1, -1)
        predicted = float(self.model.predict(X)[0])
        
        # Create contributions (importance * normalized value as proxy)
        contributions = []
        for name in self.feature_names:
            if name in features.index:
                value = float(features[name])
                imp = importance.get(name, 0)
                # Simple proxy: importance * sign of value
                contrib = imp * value if abs(value) < 100 else imp * np.sign(value)
                contributions.append(FeatureContribution(name, value, contrib))
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        
        # Identify top positive/negative
        top_positive = [c.feature_name for c in contributions if c.contribution > 0][:3]
        top_negative = [c.feature_name for c in contributions if c.contribution < 0][:3]
        
        return PredictionBreakdown(
            player_id=player_id,
            player_name=player_name,
            predicted_points=predicted,
            base_value=0.0,  # Would need SHAP for actual base value
            contributions=contributions,
            top_positive=top_positive,
            top_negative=top_negative,
        )
    
    def compare_players(
        self,
        player1_features: pd.Series,
        player2_features: pd.Series,
        player1_name: str = "Player 1",
        player2_name: str = "Player 2",
    ) -> Dict:
        """Compare predictions for two players.
        
        Highlights which features differ most between them.
        
        Returns:
            Dict with comparison details
        """
        importance = self.get_feature_importance()
        
        # Get predictions
        X1 = player1_features[self.feature_names].values.reshape(1, -1)
        X2 = player2_features[self.feature_names].values.reshape(1, -1)
        pred1 = float(self.model.predict(X1)[0])
        pred2 = float(self.model.predict(X2)[0])
        
        # Find biggest differences
        differences = []
        for name in self.feature_names:
            v1 = float(player1_features.get(name, 0))
            v2 = float(player2_features.get(name, 0))
            diff = v1 - v2
            imp = importance.get(name, 0)
            weighted_diff = diff * imp
            differences.append((name, v1, v2, diff, weighted_diff))
        
        # Sort by weighted difference
        differences.sort(key=lambda x: abs(x[4]), reverse=True)
        
        return {
            "player1": {"name": player1_name, "prediction": round(pred1, 2)},
            "player2": {"name": player2_name, "prediction": round(pred2, 2)},
            "prediction_diff": round(pred1 - pred2, 2),
            "top_differences": [
                {
                    "feature": d[0],
                    f"{player1_name}": round(d[1], 2),
                    f"{player2_name}": round(d[2], 2),
                    "diff": round(d[3], 2),
                }
                for d in differences[:5]
            ]
        }
