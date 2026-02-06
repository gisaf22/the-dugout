"""Models module - predictive models, optimization, and backtesting.

This module contains:
- Decision-specific models: captain_model, transfer_model, free_hit_model
- registry: Model lookup by decision type
- predict: Legacy prediction interface (deprecated)
- captain: Captain selection optimization  
- squad: Free Hit squad optimization (PuLP LP)
- backtest: Captain backtesting utilities
- walk_forward: Walk-forward backtesting engine

For predictions, use dugout.production.models.registry.get_model().
"""

from dugout.production.models.baseline import predict_baseline
from dugout.production.models.train_val_test import DatasetBuilder, DatasetConfig, Datasets, load_datasets
from dugout.production.models.captain import CaptainPicker, CaptainRecommendation
from dugout.production.models.squad import FreeHitOptimizer, FreeHitResult
from dugout.production.models.backtest import CaptainBacktester, CaptainBacktestSummary, CaptainGWResult
from dugout.production.models.walk_forward import (
    BacktestRunner,
    WalkForwardSummary,
    GWResult,
    RegretDiagnosis,
    PlayerPrediction,
)
from dugout.production.models.registry import get_model
from dugout.production.models.captain_model import CaptainModel
from dugout.production.models.transfer_model import TransferModel
from dugout.production.models.free_hit_model import FreeHitModel

__all__ = [
    # Baseline
    "predict_baseline",
    # Dataset splitting
    "DatasetBuilder",
    "DatasetConfig",
    "Datasets",
    "load_datasets",
    # Captain
    "CaptainPicker",
    "CaptainRecommendation",
    # Squad / Free Hit
    "FreeHitOptimizer",
    "FreeHitResult",
    # Captain Backtest
    "CaptainBacktester",
    "CaptainBacktestSummary",
    "CaptainGWResult",
    # Walk-forward Backtest
    "BacktestRunner",
    "WalkForwardSummary",
    "GWResult",
    "RegretDiagnosis",
    "PlayerPrediction",
    # Decision-specific models
    "get_model",
    "CaptainModel",
    "TransferModel",
    "FreeHitModel",
]
