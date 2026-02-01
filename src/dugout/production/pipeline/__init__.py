"""
Production Pipeline Module

End-to-end ML workflow orchestration for live predictions.

Components:
    Pipeline  - Full workflow (gather → features → split → train → evaluate)
    Trainer   - LightGBM and residual model training
    Evaluator - Model evaluation with baseline comparison
"""

from dugout.production.pipeline.runner import Pipeline
from dugout.production.pipeline.trainer import Trainer
from dugout.production.pipeline.evaluator import Evaluator, EvaluationMetrics

__all__ = [
    "Pipeline",
    "Trainer",
    "Evaluator",
    "EvaluationMetrics",
]
