"""Tests for regret analysis module."""
import pytest
from dugout.production.models.walk_forward import GWResult, PlayerPrediction, WalkForwardSummary
from dugout.production.analysis.decisions.regret_analysis import (
    RegretAnalyzer,
    RegretReport,
    GWRegretBreakdown,
    BucketStats,
)


def make_player(
    player_id: int,
    name: str,
    predicted: float,
    actual: float,
    pred_rank: int,
    actual_rank: int,
    minutes: float = 90.0,
) -> PlayerPrediction:
    """Helper to create PlayerPrediction."""
    return PlayerPrediction(
        player_id=player_id,
        player_name=name,
        position="MID",
        predicted_points=predicted,
        actual_points=actual,
        predicted_rank=pred_rank,
        actual_rank=actual_rank,
        minutes=minutes,
    )


def make_gw_result(gw: int, predictions: list) -> GWResult:
    """Helper to create GWResult with required fields."""
    return GWResult(
        gw=gw,
        n_players=len(predictions),
        mae=0.0,
        median_ae=0.0,
        rmse=0.0,
        spearman_corr=0.0,
        spearman_pval=0.0,
        top5_hit_rate=0.0,
        top10_hit_rate=0.0,
        regret_5=0.0,
        regret_10=0.0,
        player_predictions=predictions,
    )


def make_summary(gw_results: list) -> WalkForwardSummary:
    """Helper to create WalkForwardSummary."""
    return WalkForwardSummary(
        gw_results=gw_results,
        min_train_gws=5,
        test_gws=[r.gw for r in gw_results],
        total_predictions=sum(r.n_players for r in gw_results),
    )


class TestBucketStats:
    def test_repr(self):
        stats = BucketStats(
            name="Test",
            count=3,
            total_regret=15.0,
            mean_regret=5.0,
            pct_of_total_regret=0.5,
        )
        rep = repr(stats)
        assert "Test" in rep
        assert "n=3" in rep
        assert "50.0%" in rep


class TestRegretAnalyzer:
    def test_empty_summary(self):
        """Empty backtest summary returns empty report."""
        analyzer = RegretAnalyzer(k=5)
        summary = make_summary([])
        report = analyzer.analyze(summary)
        assert report.total_regret == 0.0
        assert len(report.gw_breakdowns) == 0
    
    def test_no_predictions_stored(self):
        """GWResults without player_predictions are skipped."""
        analyzer = RegretAnalyzer(k=5)
        gw_result = make_gw_result(10, [])
        summary = make_summary([gw_result])
        report = analyzer.analyze(summary)
        assert len(report.gw_breakdowns) == 0
    
    def test_perfect_predictions_no_regret(self):
        """Perfect predictions have zero regret."""
        analyzer = RegretAnalyzer(k=2)
        
        # Perfect prediction: ranks match exactly
        predictions = [
            make_player(1, "Player1", 10.0, 10.0, 1, 1),
            make_player(2, "Player2", 8.0, 8.0, 2, 2),
            make_player(3, "Player3", 5.0, 5.0, 3, 3),
            make_player(4, "Player4", 3.0, 3.0, 4, 4),
        ]
        
        gw_result = make_gw_result(10, predictions)
        summary = make_summary([gw_result])
        
        report = analyzer.analyze(summary)
        assert report.total_regret == 0.0
        assert report.gw_breakdowns[0].bucket_a.count == 0
        assert report.gw_breakdowns[0].bucket_b.count == 0
    
    def test_bucket_a_overrated(self):
        """Bucket A captures overrated picks."""
        analyzer = RegretAnalyzer(k=2)
        
        # Predicted rank 1 but actual rank 4 (overrated)
        predictions = [
            make_player(1, "Overrated", 10.0, 2.0, 1, 4),  # Overrated
            make_player(2, "Player2", 8.0, 5.0, 2, 3),
            make_player(3, "Player3", 5.0, 8.0, 3, 2),
            make_player(4, "Best", 3.0, 10.0, 4, 1),  # Actually best
        ]
        
        gw_result = make_gw_result(10, predictions)
        summary = make_summary([gw_result])
        
        report = analyzer.analyze(summary)
        
        # Bucket A should have the overrated player
        assert report.gw_breakdowns[0].bucket_a.count >= 1
        assert "Overrated" in report.gw_breakdowns[0].bucket_a.example_players
    
    def test_bucket_b_missed_ceiling(self):
        """Bucket B captures missed high scorers."""
        analyzer = RegretAnalyzer(k=2)
        
        # Player with predicted rank 3 but actual rank 1 (missed ceiling)
        predictions = [
            make_player(1, "Player1", 10.0, 5.0, 1, 3),
            make_player(2, "Player2", 8.0, 4.0, 2, 4),
            make_player(3, "Missed", 5.0, 12.0, 3, 1),  # Missed ceiling
            make_player(4, "Player4", 3.0, 6.0, 4, 2),
        ]
        
        gw_result = make_gw_result(10, predictions)
        summary = make_summary([gw_result])
        
        report = analyzer.analyze(summary)
        
        # Bucket B should have the missed player
        assert report.gw_breakdowns[0].bucket_b.count >= 1
        assert "Missed" in report.gw_breakdowns[0].bucket_b.example_players
    
    def test_bucket_c_minutes_failure(self):
        """Bucket C captures minutes failures."""
        analyzer = RegretAnalyzer(k=2, minutes_threshold=60.0)
        
        # Top predicted player only played 30 minutes
        predictions = [
            make_player(1, "Benched", 10.0, 1.0, 1, 4, minutes=30.0),  # Minutes failure
            make_player(2, "Player2", 8.0, 8.0, 2, 2),
            make_player(3, "Player3", 5.0, 5.0, 3, 3),
            make_player(4, "Player4", 3.0, 10.0, 4, 1),
        ]
        
        gw_result = make_gw_result(10, predictions)
        summary = make_summary([gw_result])
        
        report = analyzer.analyze(summary)
        
        # Bucket C should have the benched player
        assert report.gw_breakdowns[0].bucket_c.count >= 1
        assert "Benched" in report.gw_breakdowns[0].bucket_c.example_players
    
    def test_multiple_gws_aggregation(self):
        """Multiple GWs are aggregated correctly."""
        analyzer = RegretAnalyzer(k=2)
        
        predictions1 = [
            make_player(1, "P1", 10.0, 10.0, 1, 1),
            make_player(2, "P2", 8.0, 8.0, 2, 2),
            make_player(3, "P3", 5.0, 5.0, 3, 3),
        ]
        
        predictions2 = [
            make_player(1, "P1", 10.0, 5.0, 1, 3),  # Overrated in GW2
            make_player(2, "P2", 8.0, 10.0, 2, 1),
            make_player(3, "P3", 5.0, 8.0, 3, 2),
        ]
        
        gw1 = make_gw_result(10, predictions1)
        gw2 = make_gw_result(11, predictions2)
        
        summary = make_summary([gw1, gw2])
        
        report = analyzer.analyze(summary)
        
        assert len(report.gw_breakdowns) == 2
        assert report.mean_regret_per_gw > 0
    
    def test_top_offenders_tracking(self):
        """Top overrated/missed players are tracked across GWs."""
        analyzer = RegretAnalyzer(k=2)
        
        # Same player overrated in multiple GWs
        gw_results = []
        for gw in range(10, 15):
            predictions = [
                make_player(1, "ChronicOverrater", 10.0, 2.0, 1, 4),
                make_player(2, "P2", 8.0, 8.0, 2, 2),
                make_player(3, "P3", 5.0, 5.0, 3, 3),
                make_player(4, "P4", 3.0, 10.0, 4, 1),
            ]
            gw_results.append(make_gw_result(gw, predictions))
        
        summary = make_summary(gw_results)
        
        report = analyzer.analyze(summary)
        
        # ChronicOverrater should be top overrated player
        assert "ChronicOverrater" in report.top_overrated_players
    
    def test_summary_output(self):
        """Summary generates readable output."""
        analyzer = RegretAnalyzer(k=2)
        
        predictions = [
            make_player(1, "Overrated", 10.0, 2.0, 1, 4),
            make_player(2, "P2", 8.0, 8.0, 2, 2),
            make_player(3, "Missed", 5.0, 10.0, 3, 1),
            make_player(4, "P4", 3.0, 5.0, 4, 3),
        ]
        
        gw_result = make_gw_result(10, predictions)
        summary = make_summary([gw_result])
        
        report = analyzer.analyze(summary)
        summary_text = report.summary()
        
        assert "REGRET ANALYSIS REPORT" in summary_text
        assert "Bucket A" in summary_text
        assert "Bucket B" in summary_text
        assert "Bucket C" in summary_text
