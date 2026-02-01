"""
The Dugout - Decision Support for Fantasy Premier League

Three decisions: Captain, Transfer-In, Free Hit.
One rule: argmax(predicted_points)

Structure:
    production/  - App-facing predictive system
        data/      - Data access layer
        features/  - Feature engineering
        models/    - Optimization & backtesting
        pipeline/  - End-to-end workflow
    research/    - Evidence-driven, regret-evaluated system
        pipeline/  - Frozen research stages

Usage:
    from dugout.production import Pipeline
    from dugout.production.data import DataReader
    from dugout.research.pipeline import run_pipeline
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
