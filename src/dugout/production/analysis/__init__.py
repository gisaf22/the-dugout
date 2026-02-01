"""Analysis module - model and decision analysis tools.

This module provides tools for understanding predictions and decisions:

Submodules:
    models/    - Model-focused analysis (metrics, explainability)
    decisions/ - Decision-focused analysis (regret, evaluation)

Re-exports for convenience:
    - PredictionExplainer, evaluate_predictions, captain_metrics (from models)
    - DecisionEvaluator, run_decision_eval, RegretAnalyzer (from decisions)
"""

# Model analysis
from dugout.production.analysis.models.explainer import PredictionExplainer
from dugout.production.analysis.models.metrics import evaluate_predictions, captain_metrics

# Decision analysis
from dugout.production.analysis.decisions.decision_eval import DecisionEvaluator, run_decision_eval
from dugout.production.analysis.decisions.regret_analysis import (
    RegretAnalyzer,
    RegretReport,
    GWRegretBreakdown,
    BucketStats,
)

__all__ = [
    # Model analysis
    "PredictionExplainer",
    "evaluate_predictions",
    "captain_metrics",
    # Decision analysis
    "DecisionEvaluator",
    "run_decision_eval",
    "RegretAnalyzer",
    "RegretReport",
    "GWRegretBreakdown",
    "BucketStats",
]
