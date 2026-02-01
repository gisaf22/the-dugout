"""Smoke test for production decision evaluation harness.

Verifies that the decision evaluator:
1. Runs without errors on available data
2. Produces a valid JSON report with required keys
3. Returns finite numeric metrics
"""

import json
import math
from pathlib import Path

import pytest

from dugout.production.analysis.decisions import DecisionEvaluator, run_decision_eval


class TestDecisionEvalSmoke:
    """Smoke tests for decision evaluation."""
    
    def test_evaluator_runs_without_error(self):
        """Evaluator should complete without raising exceptions."""
        evaluator = DecisionEvaluator()
        result = evaluator.evaluate(verbose=False)
        
        assert result is not None
        assert result.n_gws > 0
    
    def test_result_has_required_fields(self):
        """Result should contain all required metrics."""
        evaluator = DecisionEvaluator()
        result = evaluator.evaluate(verbose=False)
        
        # Check all required fields exist
        assert hasattr(result, "gw_range")
        assert hasattr(result, "n_gws")
        assert hasattr(result, "captain_mean_regret")
        assert hasattr(result, "captain_median_regret")
        assert hasattr(result, "captain_pct_regret_ge_10")
        assert hasattr(result, "captain_total_regret")
    
    def test_metrics_are_finite_numbers(self):
        """All metrics should be finite numbers."""
        evaluator = DecisionEvaluator()
        result = evaluator.evaluate(verbose=False)
        
        assert isinstance(result.n_gws, int)
        assert result.n_gws > 0
        
        assert isinstance(result.captain_mean_regret, (int, float))
        assert math.isfinite(result.captain_mean_regret)
        
        assert isinstance(result.captain_median_regret, (int, float))
        assert math.isfinite(result.captain_median_regret)
        
        assert isinstance(result.captain_pct_regret_ge_10, (int, float))
        assert 0 <= result.captain_pct_regret_ge_10 <= 100
        
        assert isinstance(result.captain_total_regret, (int, float))
        assert math.isfinite(result.captain_total_regret)
    
    def test_gw_range_is_valid(self):
        """GW range should be a valid tuple of ints."""
        evaluator = DecisionEvaluator()
        result = evaluator.evaluate(verbose=False)
        
        assert isinstance(result.gw_range, tuple)
        assert len(result.gw_range) == 2
        assert result.gw_range[0] <= result.gw_range[1]
    
    def test_save_creates_json_file(self, tmp_path):
        """save_report should create a valid JSON file."""
        evaluator = DecisionEvaluator()
        result = evaluator.evaluate(verbose=False)
        
        # Override report path for test
        evaluator.REPORT_DIR = tmp_path
        evaluator.REPORT_PATH = tmp_path / "decision_eval.json"
        
        path = evaluator.save_report(result)
        
        assert path.exists()
        
        # Verify JSON is valid and has required keys
        with open(path) as f:
            data = json.load(f)
        
        required_keys = [
            "gw_range",
            "n_gws",
            "captain_mean_regret",
            "captain_median_regret",
            "captain_pct_regret_ge_10",
            "captain_total_regret",
        ]
        
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"
    
    def test_run_decision_eval_convenience_function(self, tmp_path, monkeypatch):
        """run_decision_eval should work end-to-end."""
        # Monkeypatch the report path to avoid overwriting real reports
        monkeypatch.setattr(
            "dugout.production.analysis.decisions.decision_eval.DecisionEvaluator.REPORT_DIR",
            tmp_path,
        )
        monkeypatch.setattr(
            "dugout.production.analysis.decisions.decision_eval.DecisionEvaluator.REPORT_PATH",
            tmp_path / "decision_eval.json",
        )
        
        result = run_decision_eval(save=True)
        
        assert result is not None
        assert (tmp_path / "decision_eval.json").exists()
