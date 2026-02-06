"""Tests for BacktestRunner walk-forward validation."""

import numpy as np
import pandas as pd
import pytest

from dugout.production.models import BacktestRunner, WalkForwardSummary, GWResult


@pytest.fixture
def sample_backtest_df():
    """Create sample DataFrame with 12 gameweeks for backtesting."""
    np.random.seed(42)
    n_players = 30
    n_gws = 12
    
    rows = []
    for gw in range(1, n_gws + 1):
        for player_id in range(1, n_players + 1):
            # Some players inactive randomly
            is_inactive = np.random.random() < 0.1
            
            rows.append({
                "player_id": player_id,
                "gw": gw,
                "target_points": np.random.randint(0, 15) if not is_inactive else 0,
                "is_inactive": is_inactive,
                # Features matching FEATURE_COLUMNS
                "per90_wmean": np.random.uniform(2, 6),
                "per90_wvar": np.random.uniform(0, 2),
                "mins_mean": np.random.uniform(60, 90),
                "appearances": np.random.randint(1, 6),
                "now_cost": np.random.uniform(40, 130),
                "total_points_season": np.random.randint(0, 100),
                "is_home_next": np.random.randint(0, 2),
                "games_since_first": np.random.randint(1, 20),
                "completed_60_count": np.random.randint(0, 6),
                "minutes_fraction": np.random.uniform(0, 1),
                "goals_sum": np.random.randint(0, 5),
                "assists_sum": np.random.randint(0, 5),
                "bonus_sum": np.random.randint(0, 10),
                "bps_mean": np.random.uniform(10, 30),
                "creativity_mean": np.random.uniform(0, 50),
                "threat_mean": np.random.uniform(0, 50),
                "influence_mean": np.random.uniform(0, 50),
                "ict_mean": np.random.uniform(0, 15),
                "xg_sum": np.random.uniform(0, 3),
                "xa_sum": np.random.uniform(0, 3),
                # Minutes risk features (Phase 2)
                "start_rate_5": np.random.uniform(0.5, 1.0),
                "mins_std_5": np.random.uniform(0, 30),
                "mins_below_60_rate_5": np.random.uniform(0, 0.5),
                # Tail risk features (Phase 2.3)
                "haul_rate_5": np.random.uniform(0, 0.3),
                # Interaction features
                "threat_x_mins": np.random.uniform(0, 4000),
                "ict_x_home": np.random.uniform(0, 15),
                "xg_x_apps": np.random.uniform(0, 15),
            })
    
    return pd.DataFrame(rows)


class TestBacktestRunner:
    """Tests for BacktestRunner."""
    
    def test_init_defaults(self):
        """Test default initialization."""
        runner = BacktestRunner()
        assert runner.min_train_gws == 5
        assert runner.gw_column == "gw"
        assert runner.target_column == "target_points"
    
    def test_run_produces_results(self, sample_backtest_df):
        """Test that backtest produces results."""
        runner = BacktestRunner(min_train_gws=5, verbose=False)
        results = runner.run(sample_backtest_df)
        
        assert isinstance(results, WalkForwardSummary)
        assert len(results.gw_results) > 0
    
    def test_train_gws_before_test(self, sample_backtest_df):
        """Test that training data is always before test GW."""
        runner = BacktestRunner(min_train_gws=5, verbose=False)
        results = runner.run(sample_backtest_df)
        
        # First test GW should be after min_train_gws
        assert results.test_gws[0] >= runner.min_train_gws + 1
    
    def test_filters_inactive_players(self, sample_backtest_df):
        """Test that inactive players are filtered before prediction."""
        runner = BacktestRunner(min_train_gws=5, verbose=False)
        
        # Make all players in GW 10 inactive
        sample_backtest_df.loc[sample_backtest_df["gw"] == 10, "is_inactive"] = True
        
        results = runner.run(sample_backtest_df, start_gw=6, end_gw=11)
        
        # GW 10 should be skipped (all inactive)
        gw_numbers = [r.gw for r in results.gw_results]
        assert 10 not in gw_numbers
    
    def test_metrics_computed(self, sample_backtest_df):
        """Test that all expected metrics are computed."""
        runner = BacktestRunner(min_train_gws=5, verbose=False)
        results = runner.run(sample_backtest_df)
        
        gw_result = results.gw_results[0]
        
        # Check all metrics exist
        assert hasattr(gw_result, "mae")
        assert hasattr(gw_result, "rmse")
        assert hasattr(gw_result, "spearman_corr")
        assert hasattr(gw_result, "top5_hit_rate")
        assert hasattr(gw_result, "top10_hit_rate")
        assert hasattr(gw_result, "regret_5")
        assert hasattr(gw_result, "regret_10")
        assert hasattr(gw_result, "mean_bias")
        assert hasattr(gw_result, "baseline_mae")
    
    def test_baseline_comparison(self, sample_backtest_df):
        """Test that baseline comparison is computed."""
        runner = BacktestRunner(min_train_gws=5, verbose=False)
        results = runner.run(sample_backtest_df)
        
        gw_result = results.gw_results[0]
        
        # mae_vs_baseline should be mae - baseline_mae
        expected_diff = gw_result.mae - gw_result.baseline_mae
        assert abs(gw_result.mae_vs_baseline - expected_diff) < 0.001
    
    def test_to_dataframe(self, sample_backtest_df):
        """Test conversion to DataFrame."""
        runner = BacktestRunner(min_train_gws=5, verbose=False)
        results = runner.run(sample_backtest_df)
        
        df = results.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert "gw" in df.columns
        assert "mae" in df.columns
        assert "spearman" in df.columns
        assert len(df) == len(results.gw_results)


class TestGWResult:
    """Tests for GWResult dataclass."""
    
    def test_gw_result_creation(self):
        """Test GWResult can be created with all fields."""
        result = GWResult(
            gw=10,
            n_players=350,
            mae=1.78,
            median_ae=1.5,
            rmse=2.34,
            spearman_corr=0.45,
            spearman_pval=0.001,
            top5_hit_rate=0.6,
            top10_hit_rate=0.5,
            regret_5=15.0,
            regret_10=25.0,
            mean_bias=0.1,
            baseline_mae=2.0,
            mae_vs_baseline=-0.22,
        )
        
        assert result.gw == 10
        assert result.mae == 1.78
        assert result.top5_hit_rate == 0.6


class TestWalkForwardSummary:
    """Tests for WalkForwardSummary."""
    
    @pytest.fixture
    def sample_gw_results(self):
        """Create sample GW results."""
        return [
            GWResult(6, 300, mae=1.5, median_ae=1.2, rmse=2.0,
                     spearman_corr=0.4, spearman_pval=0.01,
                     top5_hit_rate=0.6, top10_hit_rate=0.5,
                     regret_5=10, regret_10=20, mean_bias=0.1,
                     baseline_mae=2.0, mae_vs_baseline=-0.5),
            GWResult(7, 305, mae=1.8, median_ae=1.5, rmse=2.3,
                     spearman_corr=0.5, spearman_pval=0.01,
                     top5_hit_rate=0.4, top10_hit_rate=0.6,
                     regret_5=15, regret_10=25, mean_bias=-0.1,
                     baseline_mae=2.1, mae_vs_baseline=-0.3),
            GWResult(8, 310, mae=1.6, median_ae=1.3, rmse=2.1,
                     spearman_corr=0.45, spearman_pval=0.01,
                     top5_hit_rate=0.8, top10_hit_rate=0.7,
                     regret_5=5, regret_10=15, mean_bias=0.0,
                     baseline_mae=1.9, mae_vs_baseline=-0.3),
        ]
    
    def test_mean_mae(self, sample_gw_results):
        """Test mean MAE calculation."""
        summary = WalkForwardSummary(
            gw_results=sample_gw_results,
            min_train_gws=5,
            test_gws=[6, 7, 8],
            total_predictions=915,
        )
        
        expected = np.mean([1.5, 1.8, 1.6])
        assert abs(summary.mean_mae - expected) < 0.001
    
    def test_mean_spearman(self, sample_gw_results):
        """Test mean Spearman calculation."""
        summary = WalkForwardSummary(
            gw_results=sample_gw_results,
            min_train_gws=5,
            test_gws=[6, 7, 8],
            total_predictions=915,
        )
        
        expected = np.mean([0.4, 0.5, 0.45])
        assert abs(summary.mean_spearman - expected) < 0.001
    
    def test_improvement_vs_baseline(self, sample_gw_results):
        """Test improvement vs baseline is positive when model is better."""
        summary = WalkForwardSummary(
            gw_results=sample_gw_results,
            min_train_gws=5,
            test_gws=[6, 7, 8],
            total_predictions=915,
        )
        
        # All results have negative mae_vs_baseline (model better)
        assert summary.mean_improvement_vs_baseline > 0
    
    def test_print_summary(self, sample_gw_results, capsys):
        """Test print_summary outputs correctly."""
        summary = WalkForwardSummary(
            gw_results=sample_gw_results,
            min_train_gws=5,
            test_gws=[6, 7, 8],
            total_predictions=915,
        )
        summary.print_summary()
        
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out
        assert "MAE:" in captured.out
        assert "Spearman:" in captured.out
        assert "Top-5 hit:" in captured.out
        assert "Baseline" in captured.out
