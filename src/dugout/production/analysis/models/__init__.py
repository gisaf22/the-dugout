"""Model analysis - interpretability and prediction metrics.

This module provides tools for understanding model predictions:
- explainer: Feature importance and prediction breakdown
- metrics: Evaluation metrics for predictions (MAE, RMSE, R², etc.)
"""

from dugout.production.analysis.models.explainer import PredictionExplainer
from dugout.production.analysis.models.metrics import evaluate_predictions, captain_metrics

__all__ = [
    "PredictionExplainer",
    "evaluate_predictions",
    "captain_metrics",
]
