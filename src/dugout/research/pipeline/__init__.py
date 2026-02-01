"""
Research Pipeline Module — Evidence-Driven, Regret-Evaluated Stages

This module contains the frozen research pipeline stages that produce
evidence for policy decisions. All stages are deterministic and reproducible.

Stages:
    Stage 2:  targets.py               → targets.csv
    Stage 3:  features_participation.py → features_participation.csv
    Stage 4a: features_performance.py   → features_performance.csv
    Stage 4b: features_fixture_context.py → features_fixture_context.csv
    Stage 5:  belief_models.py          → beliefs.csv, models/*.pkl
    Stage 5b: belief_models_fixture.py  → beliefs_fixture_context.csv
    Stage 6a: stage_6a_captain_policy.py → evaluation_captain.csv
    Stage 6b: stage_6b_captain_baselines.py → evaluation_captain_baselines.csv
    Stage 6c: stage_6c_captain_revision.py → evaluation_captain_revised.csv
    Stage 7a: stage_7a_transfer_in.py    → evaluation_transfer_in.csv
    Stage 8a: stage_8a_multigw_beliefs.py → beliefs_multigw.csv
    Stage 8b: stage_8b_multigw_hold.py   → evaluation_multigw_hold.csv

Runner:
    run_all.py       - Sequential stage orchestrator
    report_builder.py - Metrics collection and JSON report generation

Usage:
    python -m dugout.research.pipeline.run_all

Output:
    storage/reports/research_report.json
"""

from dugout.research.pipeline.run_all import run_pipeline
from dugout.research.pipeline.report_builder import build_report

__all__ = [
    "run_pipeline",
    "build_report",
]
