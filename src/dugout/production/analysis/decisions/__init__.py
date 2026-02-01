"""Decision analysis - evaluation and regret diagnostics.

This module provides tools for analyzing decision quality:
- decision_eval: Production decision evaluation harness
- regret_analysis: Post-hoc regret source analysis
"""

from dugout.production.analysis.decisions.decision_eval import DecisionEvaluator, run_decision_eval
from dugout.production.analysis.decisions.regret_analysis import (
    RegretAnalyzer,
    RegretReport,
    GWRegretBreakdown,
    BucketStats,
)

__all__ = [
    "DecisionEvaluator",
    "run_decision_eval",
    "RegretAnalyzer",
    "RegretReport",
    "GWRegretBreakdown",
    "BucketStats",
]
