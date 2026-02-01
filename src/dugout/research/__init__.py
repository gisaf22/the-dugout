"""
Research Pipeline — Evidence-Driven, Regret-Evaluated System

This module contains the research pipeline for policy evaluation using
regret-based metrics. All stages are frozen, deterministic, and reproducible.

The research pipeline answers: "Does this belief or policy reduce regret?"

Structure:
    pipeline/           - All research stages (targets, features, beliefs, policies)
    pipeline/run_all.py - Sequential stage orchestrator
    pipeline/report_builder.py - Metrics collection and JSON report generation

Usage:
    python -m dugout.research.pipeline.run_all

Output:
    storage/datasets/*.csv   - Frozen research artifacts
    storage/reports/*.json   - Evidence reports

See Also:
    dugout.production - Predictive, app-facing system
"""

from dugout.research.pipeline.run_all import run_pipeline
from dugout.research.pipeline.report_builder import build_report

__all__ = ["run_pipeline", "build_report"]
