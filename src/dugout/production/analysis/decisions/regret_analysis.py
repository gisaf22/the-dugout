"""
Post-backtest regret analysis module.

Diagnoses WHY the model loses decision-level value (regret) without
modifying core backtesting logic. Consumes WalkForwardSummary outputs.

Responsibility:
- This module EXPLAINS errors post-hoc
- It does not alter backtest logic or make predictions
- It consumes backtest outputs (WalkForwardSummary, GWResult, PlayerPrediction)

Regret Buckets:
- Bucket A (Overrated): Predicted top-K but not actual top-K
- Bucket B (Missed Ceiling): Predicted rank 6-10 but actual top-K  
- Bucket C (Minutes Failure): Any bucket where minutes < 60
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from dugout.production.models.walk_forward import WalkForwardSummary, GWResult, PlayerPrediction


@dataclass
class BucketStats:
    """Statistics for a single regret bucket."""
    
    name: str
    count: int = 0
    total_regret: float = 0.0
    mean_regret: float = 0.0
    pct_of_total_regret: float = 0.0
    example_players: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return (
            f"BucketStats({self.name}: n={self.count}, "
            f"regret={self.total_regret:.1f}, "
            f"mean={self.mean_regret:.2f}, "
            f"pct={self.pct_of_total_regret:.1%})"
        )


@dataclass
class GWRegretBreakdown:
    """Regret breakdown for a single gameweek."""
    
    gw: int
    total_regret: float
    bucket_a: BucketStats  # Overrated picks
    bucket_b: BucketStats  # Missed ceilings
    bucket_c: BucketStats  # Minutes failures (cross-cut)
    
    def __repr__(self) -> str:
        return f"GWRegretBreakdown(gw={self.gw}, regret={self.total_regret:.1f})"


@dataclass
class RegretReport:
    """Full regret analysis report across all gameweeks."""
    
    gw_breakdowns: List[GWRegretBreakdown]
    
    # Aggregated stats
    total_regret: float = 0.0
    mean_regret_per_gw: float = 0.0
    
    # Bucket aggregates
    bucket_a_total: float = 0.0  # Overrated contribution
    bucket_b_total: float = 0.0  # Missed ceiling contribution
    bucket_c_total: float = 0.0  # Minutes failures contribution
    
    bucket_a_pct: float = 0.0
    bucket_b_pct: float = 0.0
    bucket_c_pct: float = 0.0
    
    # Top offenders (players appearing most in bucket A)
    top_overrated_players: List[str] = field(default_factory=list)
    top_missed_players: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Human-readable summary of regret sources."""
        lines = [
            "=" * 50,
            "REGRET ANALYSIS REPORT",
            "=" * 50,
            f"Total Regret: {self.total_regret:.1f} pts across {len(self.gw_breakdowns)} GWs",
            f"Mean Regret/GW: {self.mean_regret_per_gw:.1f} pts",
            "",
            "REGRET SOURCES:",
            f"  Bucket A (Overrated picks):   {self.bucket_a_total:6.1f} pts ({self.bucket_a_pct:5.1%})",
            f"  Bucket B (Missed ceilings):   {self.bucket_b_total:6.1f} pts ({self.bucket_b_pct:5.1%})",
            f"  Bucket C (Minutes failures):  {self.bucket_c_total:6.1f} pts ({self.bucket_c_pct:5.1%})",
            "",
        ]
        
        if self.top_overrated_players:
            lines.append("TOP OVERRATED PLAYERS (most false positives):")
            for p in self.top_overrated_players[:5]:
                lines.append(f"  - {p}")
            lines.append("")
            
        if self.top_missed_players:
            lines.append("TOP MISSED PLAYERS (most false negatives):")
            for p in self.top_missed_players[:5]:
                lines.append(f"  - {p}")
            lines.append("")
            
        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class BucketAProfile:
    """Profile of Bucket A (overrated) players - what makes them look good but underperform?"""
    
    # Sample size
    n_players: int = 0
    n_unique_players: int = 0
    
    # Position breakdown
    position_counts: Dict[str, int] = field(default_factory=dict)
    position_pct: Dict[str, float] = field(default_factory=dict)
    
    # Feature averages for Bucket A players
    avg_mins_mean: float = 0.0
    avg_xg_sum: float = 0.0
    avg_xa_sum: float = 0.0
    avg_threat_mean: float = 0.0
    avg_creativity_mean: float = 0.0
    avg_ict_mean: float = 0.0
    avg_bonus_sum: float = 0.0
    avg_start_rate_5: float = 0.0
    avg_mins_std_5: float = 0.0
    
    # Prediction gap
    avg_predicted_points: float = 0.0
    avg_actual_points: float = 0.0
    avg_overestimate: float = 0.0  # predicted - actual
    
    # Comparison to actual top-K (what we missed)
    topk_avg_xg_sum: float = 0.0
    topk_avg_threat_mean: float = 0.0
    topk_avg_ict_mean: float = 0.0
    
    def summary(self) -> str:
        """Human-readable Bucket A profile."""
        lines = [
            "",
            "=" * 60,
            "BUCKET A PROFILE (Overrated Players)",
            "=" * 60,
            f"Sample: {self.n_players} picks ({self.n_unique_players} unique players)",
            "",
            "POSITION BREAKDOWN:",
        ]
        
        for pos, pct in sorted(self.position_pct.items(), key=lambda x: -x[1]):
            pos_name = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(int(pos) if pos.isdigit() else 0, pos)
            lines.append(f"  {pos_name}: {self.position_counts.get(pos, 0)} ({pct:.0%})")
        
        lines.extend([
            "",
            "FEATURE PROFILE (Bucket A avg vs what we should have picked):",
            f"  Minutes:     {self.avg_mins_mean:5.1f} (reliable)",
            f"  xG (5-game): {self.avg_xg_sum:5.2f} vs Top-K: {self.topk_avg_xg_sum:5.2f}",
            f"  Threat:      {self.avg_threat_mean:5.1f} vs Top-K: {self.topk_avg_threat_mean:5.1f}",
            f"  ICT:         {self.avg_ict_mean:5.1f} vs Top-K: {self.topk_avg_ict_mean:5.1f}",
            f"  Creativity:  {self.avg_creativity_mean:5.1f}",
            f"  Bonus:       {self.avg_bonus_sum:5.1f}",
            f"  Start Rate:  {self.avg_start_rate_5:5.2f}",
            "",
            "PREDICTION GAP:",
            f"  Avg Predicted: {self.avg_predicted_points:.2f}",
            f"  Avg Actual:    {self.avg_actual_points:.2f}",
            f"  Overestimate:  {self.avg_overestimate:+.2f}",
            "",
            "INTERPRETATION:",
        ])
        
        # Auto-interpretation
        if self.avg_xg_sum < self.topk_avg_xg_sum * 0.7:
            lines.append("  ⚠️ Low xG signal: picking safe, low-ceiling players")
        if self.avg_threat_mean < self.topk_avg_threat_mean * 0.7:
            lines.append("  ⚠️ Low threat: not prioritizing attacking involvement")
        if self.avg_mins_mean > 80 and self.avg_start_rate_5 > 0.9:
            lines.append("  ⚠️ High minutes + reliability: valuing floor over ceiling")
        if self.position_pct.get("2", 0) > 0.4:
            lines.append("  ⚠️ Defender-heavy: defensive assets rarely haul big")
        if self.avg_overestimate > 1.5:
            lines.append("  ⚠️ Systematic over-prediction: model expects more than delivered")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class RegretAnalyzer:
    """
    Analyzes regret sources from backtest results.
    
    Requires backtest to be run with store_predictions=True.
    """
    
    def __init__(self, k: int = 5, minutes_threshold: float = 60.0):
        """
        Args:
            k: Top-K for regret calculation (default 5)
            minutes_threshold: Minutes below which counts as "minutes failure"
        """
        self.k = k
        self.minutes_threshold = minutes_threshold
    
    def analyze(self, summary: WalkForwardSummary) -> RegretReport:
        """
        Analyze regret sources from backtest summary.
        
        Args:
            summary: WalkForwardSummary with player predictions stored
            
        Returns:
            RegretReport with bucket breakdowns
        """
        gw_breakdowns = []
        overrated_counts: Dict[str, int] = {}
        missed_counts: Dict[str, int] = {}
        
        for gw_result in summary.gw_results:
            if not gw_result.player_predictions:
                continue
                
            breakdown = self._analyze_gw(gw_result)
            gw_breakdowns.append(breakdown)
            
            # Track player occurrences for top offenders
            for pred in gw_result.player_predictions:
                if pred.predicted_rank <= self.k and pred.actual_rank > self.k:
                    name = pred.player_name or f"player_{pred.player_id}"
                    overrated_counts[name] = overrated_counts.get(name, 0) + 1
                    
                if pred.predicted_rank > self.k and pred.actual_rank <= self.k:
                    name = pred.player_name or f"player_{pred.player_id}"
                    missed_counts[name] = missed_counts.get(name, 0) + 1
        
        if not gw_breakdowns:
            return RegretReport(gw_breakdowns=[])
        
        # Aggregate stats
        total_regret = sum(b.total_regret for b in gw_breakdowns)
        bucket_a_total = sum(b.bucket_a.total_regret for b in gw_breakdowns)
        bucket_b_total = sum(b.bucket_b.total_regret for b in gw_breakdowns)
        bucket_c_total = sum(b.bucket_c.total_regret for b in gw_breakdowns)
        
        # Sort top offenders
        top_overrated = sorted(overrated_counts.items(), key=lambda x: -x[1])
        top_missed = sorted(missed_counts.items(), key=lambda x: -x[1])
        
        return RegretReport(
            gw_breakdowns=gw_breakdowns,
            total_regret=total_regret,
            mean_regret_per_gw=total_regret / len(gw_breakdowns),
            bucket_a_total=bucket_a_total,
            bucket_b_total=bucket_b_total,
            bucket_c_total=bucket_c_total,
            bucket_a_pct=bucket_a_total / total_regret if total_regret > 0 else 0.0,
            bucket_b_pct=bucket_b_total / total_regret if total_regret > 0 else 0.0,
            bucket_c_pct=bucket_c_total / total_regret if total_regret > 0 else 0.0,
            top_overrated_players=[p[0] for p in top_overrated[:10]],
            top_missed_players=[p[0] for p in top_missed[:10]],
        )
    
    def _analyze_gw(self, gw_result: GWResult) -> GWRegretBreakdown:
        """Analyze regret for a single gameweek."""
        predictions = gw_result.player_predictions
        
        # Get actual top-K points for regret calculation
        sorted_by_actual = sorted(predictions, key=lambda p: -p.actual_points)
        actual_top_k_total = sum(p.actual_points for p in sorted_by_actual[:self.k])
        
        # Get predicted top-K points (what we picked)
        sorted_by_pred = sorted(predictions, key=lambda p: -p.predicted_points)
        pred_top_k = sorted_by_pred[:self.k]
        pred_top_k_actual = sum(p.actual_points for p in pred_top_k)
        
        total_regret = actual_top_k_total - pred_top_k_actual
        
        # Bucket A: Overrated (predicted top-K, actual not top-K)
        bucket_a_players = [
            p for p in predictions
            if p.predicted_rank <= self.k and p.actual_rank > self.k
        ]
        bucket_a_regret = sum(
            # Regret = what we could have had - what we got
            # For overrated: we picked them but better options existed
            max(0, sorted_by_actual[min(p.predicted_rank - 1, self.k - 1)].actual_points - p.actual_points)
            for p in bucket_a_players
        ) if bucket_a_players else 0.0
        
        # Bucket B: Missed ceiling (predicted rank 6-10, actual top-K)
        bucket_b_players = [
            p for p in predictions
            if self.k < p.predicted_rank <= self.k * 2 and p.actual_rank <= self.k
        ]
        bucket_b_regret = sum(
            # We missed these players - their actual minus our K-th pick's actual
            max(0, p.actual_points - pred_top_k[-1].actual_points if pred_top_k else 0)
            for p in bucket_b_players
        ) if bucket_b_players else 0.0
        
        # Bucket C: Minutes failures (cross-cut - overlap with A/B where minutes < threshold)
        bucket_c_players = [
            p for p in predictions
            if p.predicted_rank <= self.k and p.minutes < self.minutes_threshold
        ]
        bucket_c_regret = sum(
            max(0, sorted_by_actual[min(p.predicted_rank - 1, self.k - 1)].actual_points - p.actual_points)
            for p in bucket_c_players
        ) if bucket_c_players else 0.0
        
        return GWRegretBreakdown(
            gw=gw_result.gw,
            total_regret=total_regret,
            bucket_a=BucketStats(
                name="Overrated",
                count=len(bucket_a_players),
                total_regret=bucket_a_regret,
                mean_regret=bucket_a_regret / len(bucket_a_players) if bucket_a_players else 0.0,
                pct_of_total_regret=bucket_a_regret / total_regret if total_regret > 0 else 0.0,
                example_players=[p.player_name for p in bucket_a_players[:3]],
            ),
            bucket_b=BucketStats(
                name="Missed Ceiling",
                count=len(bucket_b_players),
                total_regret=bucket_b_regret,
                mean_regret=bucket_b_regret / len(bucket_b_players) if bucket_b_players else 0.0,
                pct_of_total_regret=bucket_b_regret / total_regret if total_regret > 0 else 0.0,
                example_players=[p.player_name for p in bucket_b_players[:3]],
            ),
            bucket_c=BucketStats(
                name="Minutes Failure",
                count=len(bucket_c_players),
                total_regret=bucket_c_regret,
                mean_regret=bucket_c_regret / len(bucket_c_players) if bucket_c_players else 0.0,
                pct_of_total_regret=bucket_c_regret / total_regret if total_regret > 0 else 0.0,
                example_players=[p.player_name for p in bucket_c_players[:3]],
            ),
        )

    def profile_bucket_a(self, summary: WalkForwardSummary) -> BucketAProfile:
        """
        Build a detailed profile of Bucket A (overrated) players.
        
        Answers: What makes these players look good but underperform?
        
        Args:
            summary: WalkForwardSummary with player predictions
            
        Returns:
            BucketAProfile with feature averages and position breakdown
        """
        bucket_a_players = []
        actual_topk_players = []
        unique_names = set()
        
        for gw_result in summary.gw_results:
            if not gw_result.player_predictions:
                continue
            
            predictions = gw_result.player_predictions
            
            # Collect Bucket A players (predicted top-K, actual not top-K)
            for p in predictions:
                if p.predicted_rank <= self.k and p.actual_rank > self.k:
                    bucket_a_players.append(p)
                    unique_names.add(p.player_name)
            
            # Collect actual top-K for comparison
            sorted_by_actual = sorted(predictions, key=lambda p: -p.actual_points)
            actual_topk_players.extend(sorted_by_actual[:self.k])
        
        if not bucket_a_players:
            return BucketAProfile()
        
        n = len(bucket_a_players)
        
        # Position breakdown
        position_counts: Dict[str, int] = {}
        for p in bucket_a_players:
            pos = str(p.position)
            position_counts[pos] = position_counts.get(pos, 0) + 1
        position_pct = {pos: count / n for pos, count in position_counts.items()}
        
        # Feature averages for Bucket A
        avg_mins = sum(p.minutes for p in bucket_a_players) / n
        avg_xg = sum(p.xg_sum for p in bucket_a_players) / n
        avg_xa = sum(p.xa_sum for p in bucket_a_players) / n
        avg_threat = sum(p.threat_mean for p in bucket_a_players) / n
        avg_creativity = sum(p.creativity_mean for p in bucket_a_players) / n
        avg_ict = sum(p.ict_mean for p in bucket_a_players) / n
        avg_bonus = sum(p.bonus_sum for p in bucket_a_players) / n
        avg_start_rate = sum(p.start_rate_5 for p in bucket_a_players) / n
        avg_mins_std = sum(p.mins_std_5 for p in bucket_a_players) / n
        
        # Prediction gap
        avg_pred = sum(p.predicted_points for p in bucket_a_players) / n
        avg_actual = sum(p.actual_points for p in bucket_a_players) / n
        
        # Actual top-K averages for comparison
        n_topk = len(actual_topk_players)
        topk_avg_xg = sum(p.xg_sum for p in actual_topk_players) / n_topk if n_topk else 0
        topk_avg_threat = sum(p.threat_mean for p in actual_topk_players) / n_topk if n_topk else 0
        topk_avg_ict = sum(p.ict_mean for p in actual_topk_players) / n_topk if n_topk else 0
        
        return BucketAProfile(
            n_players=n,
            n_unique_players=len(unique_names),
            position_counts=position_counts,
            position_pct=position_pct,
            avg_mins_mean=avg_mins,
            avg_xg_sum=avg_xg,
            avg_xa_sum=avg_xa,
            avg_threat_mean=avg_threat,
            avg_creativity_mean=avg_creativity,
            avg_ict_mean=avg_ict,
            avg_bonus_sum=avg_bonus,
            avg_start_rate_5=avg_start_rate,
            avg_mins_std_5=avg_mins_std,
            avg_predicted_points=avg_pred,
            avg_actual_points=avg_actual,
            avg_overestimate=avg_pred - avg_actual,
            topk_avg_xg_sum=topk_avg_xg,
            topk_avg_threat_mean=topk_avg_threat,
            topk_avg_ict_mean=topk_avg_ict,
        )
