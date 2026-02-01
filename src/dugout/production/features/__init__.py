"""Feature engineering module.

Public API:
    FeatureBuilder - Build features from raw data
    FEATURE_COLUMNS - List of feature column names

Usage:
    from dugout.production.features import FeatureBuilder
    
    builder = FeatureBuilder()
    features = builder.build_for_player(player_history)
"""

from dugout.production.features.builder import FeatureBuilder
from dugout.production.features.definitions import (
    FEATURE_COLUMNS,
    FeatureConfig,
)
from dugout.production.features.wrangler import Wrangler

__all__ = [
    # Core API
    "FeatureBuilder",
    "FEATURE_COLUMNS",
    "FeatureConfig",
    # Data preparation
    "Wrangler",
]
