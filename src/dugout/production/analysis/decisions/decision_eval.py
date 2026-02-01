"""Production decision evaluation harness.

Computes decision-level metrics (captain regret) from production predictions
and writes a deterministic JSON report for CI/regression testing.

Usage:
    PYTHONPATH=src python -m dugout.production.analysis.decisions.decision_eval

Output:
    storage/production/reports/decision_eval.json
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from dugout.production.config import DEFAULT_DB_PATH, STORAGE_DIR
from dugout.production.data import DataReader
from dugout.production.features import FeatureBuilder
from dugout.production.models.backtest import CaptainBacktester


@dataclass
class DecisionEvalResult:
    """Decision evaluation result (deterministic, no timestamps)."""
    
    gw_range: Tuple[int, int]
    n_gws: int
    captain_mean_regret: float
    captain_median_regret: float
    captain_pct_regret_ge_10: float
    captain_total_regret: float
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "gw_range": [int(self.gw_range[0]), int(self.gw_range[1])],
            "n_gws": int(self.n_gws),
            "captain_mean_regret": round(float(self.captain_mean_regret), 4),
            "captain_median_regret": round(float(self.captain_median_regret), 4),
            "captain_pct_regret_ge_10": round(float(self.captain_pct_regret_ge_10), 4),
            "captain_total_regret": round(float(self.captain_total_regret), 4),
        }


class DecisionEvaluator:
    """Evaluates production decision quality via captain regret.
    
    Uses existing CaptainBacktester to compute per-GW regret, then
    aggregates into decision-level metrics.
    """
    
    REPORT_DIR = STORAGE_DIR / "reports"
    REPORT_PATH = REPORT_DIR / "decision_eval.json"
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize evaluator.
        
        Args:
            db_path: Path to SQLite database. Defaults to production DB.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
    
    def evaluate(
        self,
        min_gw: Optional[int] = None,
        max_gw: Optional[int] = None,
        verbose: bool = True,
    ) -> DecisionEvalResult:
        """Run decision evaluation.
        
        Args:
            min_gw: Minimum gameweek (inclusive). Defaults to first available.
            max_gw: Maximum gameweek (inclusive). Defaults to last completed.
            verbose: Print progress.
            
        Returns:
            DecisionEvalResult with aggregated metrics.
        """
        # Load data
        if verbose:
            print(f"[Production Decision Eval]")
            print(f"Database: {self.db_path}")
        
        reader = DataReader(self.db_path)
        raw_df = reader.get_all_gw_data()
        
        # Build features
        builder = FeatureBuilder()
        feature_df = builder.build_training_set(raw_df)
        
        # Filter to GW range if specified
        available_gws = sorted(feature_df["gw"].unique())
        actual_min = min_gw if min_gw else available_gws[0]
        actual_max = max_gw if max_gw else available_gws[-1]
        
        # Run backtest using frozen rule: argmax(predicted_points)
        backtester = CaptainBacktester(min_train_gws=5)
        summary = backtester.run(
            feature_df,
            squad_size=15,
            verbose=False,
        )
        
        # Extract per-GW regrets (regret is already doubled in backtest)
        # Convert to single points for consistency with research
        regrets = [r.regret / 2 for r in summary.gw_results]
        
        if not regrets:
            raise ValueError("No gameweeks evaluated - insufficient data")
        
        # Filter to requested GW range
        gw_regrets = [
            (r.gw, r.regret / 2) 
            for r in summary.gw_results 
            if actual_min <= r.gw <= actual_max
        ]
        
        if not gw_regrets:
            raise ValueError(f"No gameweeks in range {actual_min}-{actual_max}")
        
        regrets = [reg for _, reg in gw_regrets]
        gws_evaluated = [gw for gw, _ in gw_regrets]
        
        # Compute metrics
        result = DecisionEvalResult(
            gw_range=(min(gws_evaluated), max(gws_evaluated)),
            n_gws=len(regrets),
            captain_mean_regret=statistics.mean(regrets),
            captain_median_regret=statistics.median(regrets),
            captain_pct_regret_ge_10=sum(1 for r in regrets if r >= 10) / len(regrets) * 100,
            captain_total_regret=sum(regrets),
        )
        
        if verbose:
            print(f"GW: {result.gw_range[0]}–{result.gw_range[1]} (n={result.n_gws})")
            print(f"Mean regret: {result.captain_mean_regret:.2f}")
            print(f"Median regret: {result.captain_median_regret:.2f}")
            print(f"% regret >= 10: {result.captain_pct_regret_ge_10:.1f}%")
            print(f"Total regret: {result.captain_total_regret:.1f}")
        
        return result
    
    def save_report(self, result: DecisionEvalResult) -> Path:
        """Save evaluation result to JSON.
        
        Args:
            result: Evaluation result to save.
            
        Returns:
            Path to saved report.
        """
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(self.REPORT_PATH, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return self.REPORT_PATH


def run_decision_eval(
    min_gw: Optional[int] = None,
    max_gw: Optional[int] = None,
    save: bool = True,
) -> DecisionEvalResult:
    """Run decision evaluation and optionally save report.
    
    Args:
        min_gw: Minimum gameweek (inclusive).
        max_gw: Maximum gameweek (inclusive).
        save: Whether to save JSON report.
        
    Returns:
        DecisionEvalResult with metrics.
    """
    evaluator = DecisionEvaluator()
    result = evaluator.evaluate(min_gw=min_gw, max_gw=max_gw)
    
    if save:
        path = evaluator.save_report(result)
        print(f"Saved: {path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production decision evaluation")
    parser.add_argument("--min-gw", type=int, help="Minimum gameweek")
    parser.add_argument("--max-gw", type=int, help="Maximum gameweek")
    parser.add_argument("--no-save", action="store_true", help="Skip saving JSON report")
    
    args = parser.parse_args()
    
    run_decision_eval(
        min_gw=args.min_gw,
        max_gw=args.max_gw,
        save=not args.no_save,
    )
