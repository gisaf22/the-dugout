"""
Production Pipeline — Predictive, App-Facing System

This module contains the production pipeline for live predictions and app support.
It focuses on MAE/RMSE-optimized point predictions, not regret-evaluated decisions.

Components:
    Pipeline - End-to-end workflow (gather → features → split → train → evaluate)
    Trainer  - LightGBM and residual model training
    Evaluator - Model evaluation with baseline comparison

Usage:
    from dugout.production import Pipeline
    
    # Full pipeline
    Pipeline.run()

See Also:
    dugout.research - Evidence-driven, regret-evaluated system
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
