"""Walk-forward backtesting for FPL points prediction.

Simulates real weekly FPL deployment with honest, leakage-free evaluation.

⚠️  DIAGNOSTIC EXCEPTION: This module contains lgb.train() calls.
    This is intentional and NOT a violation of the canonical training path.
    
    WHY: Walk-forward validation requires training a fresh model for each
    test gameweek (train on GW 1..t-1, predict GW t). This is fundamentally
    different from production training which trains once and deploys.
    
    RULE: Models trained here are NEVER saved or used for production.
    Production models MUST be trained via dugout.production.pipeline.trainer.

Why Walk-Forward Validation?
    Standard cross-validation shuffles data randomly, allowing models to "peek"
    at future information. In FPL, we predict GW t using only data from GW < t.
    Walk-forward validation respects this temporal constraint.

How Leakage is Prevented:
    1. For each test GW t, we train only on GWs < t
    2. Features are pre-computed with shift(1) to avoid same-GW leakage
    3. Availability filtering happens BEFORE prediction (mirrors real decisions)

Availability Handling:
    Players with is_inactive=True are excluded before prediction, matching
    the real constraint that you can't pick injured/suspended players.

Decision Metrics:
    - MAE/RMSE: Raw prediction accuracy
    - Spearman correlation: Ranking quality (critical for captain/transfer picks)
    - Top-K hit rate: How often our top K predictions include actual top K scorers
    - Regret@K: Points left on table vs optimal Top-K selection

Key Classes:
    BacktestRunner - Walk-forward backtesting with decision metrics

Usage:
    from dugout.production.models import BacktestRunner
    
    runner = BacktestRunner(min_train_gws=5)
    results = runner.run(df)
    results.print_summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dugout.production.features.definitions import FEATURE_COLUMNS
from dugout.production.models.baseline import predict_baseline


@dataclass
class RegretDiagnosis:
    """Breakdown of regret sources for a single GW."""
    # False positives: predicted top-5, actual outside top-5
    false_positive_count: int  # How many of our top-5 picks weren't actually top-5
    false_positive_regret: float  # Points lost from picking wrong players
    
    # False negatives: actual top-5, predicted outside top-5
    false_negative_count: int  # How many actual top-5 we missed
    missed_from_rank_6_10: int  # Actual top-5 we ranked 6-10 (near misses)
    missed_from_rank_11_plus: int  # Actual top-5 we ranked 11+ (big misses)
    
    # Ceiling vs floor errors
    top_scorer_predicted_rank: int  # Where we ranked the actual #1 scorer
    top_scorer_points: float
    our_top_pick_actual_rank: int  # Actual rank of who we predicted #1
    our_top_pick_points: float


@dataclass
class PlayerPrediction:
    """Single player prediction for regret analysis."""
    player_id: int
    player_name: str
    position: str
    predicted_points: float
    actual_points: float
    predicted_rank: int
    actual_rank: int
    minutes: float  # mins_mean (expected minutes proxy)
    
    # Diagnostic features for Bucket A analysis
    xg_sum: float = 0.0
    xa_sum: float = 0.0
    threat_mean: float = 0.0
    creativity_mean: float = 0.0
    ict_mean: float = 0.0
    bonus_sum: float = 0.0
    start_rate_5: float = 0.0
    mins_std_5: float = 0.0


@dataclass
class GWResult:
    """Results from a single gameweek backtest."""
    gw: int
    n_players: int
    
    # Core accuracy
    mae: float
    median_ae: float
    rmse: float
    
    # Ranking metrics
    spearman_corr: float
    spearman_pval: float
    
    # Decision metrics
    top5_hit_rate: float  # Proportion of actual top-5 in predicted top-5
    top10_hit_rate: float
    regret_5: float  # Points missed vs optimal top-5
    regret_10: float
    
    # Regret diagnosis
    regret_diagnosis: Optional[RegretDiagnosis] = None
    
    # Player-level predictions (for regret analysis module)
    player_predictions: Optional[List[PlayerPrediction]] = None
    
    # Bias
    mean_bias: float = 0.0  # Mean(predicted - actual), positive = over-predict
    
    # Baseline comparison
    baseline_mae: float = 0.0
    mae_vs_baseline: float = 0.0  # Negative = model is better


@dataclass
class WalkForwardSummary:
    """Aggregate results from full backtest."""
    gw_results: List[GWResult]
    
    # Training info
    min_train_gws: int
    test_gws: List[int]
    total_predictions: int
    
    @property
    def mean_mae(self) -> float:
        return np.mean([r.mae for r in self.gw_results])
    
    @property
    def std_mae(self) -> float:
        return np.std([r.mae for r in self.gw_results])
    
    @property
    def mean_rmse(self) -> float:
        return np.mean([r.rmse for r in self.gw_results])
    
    @property
    def mean_spearman(self) -> float:
        return np.mean([r.spearman_corr for r in self.gw_results])
    
    @property
    def mean_top5_hit_rate(self) -> float:
        return np.mean([r.top5_hit_rate for r in self.gw_results])
    
    @property
    def mean_top10_hit_rate(self) -> float:
        return np.mean([r.top10_hit_rate for r in self.gw_results])
    
    @property
    def mean_regret_5(self) -> float:
        return np.mean([r.regret_5 for r in self.gw_results])
    
    @property
    def mean_regret_10(self) -> float:
        return np.mean([r.regret_10 for r in self.gw_results])
    
    @property
    def mean_bias(self) -> float:
        return np.mean([r.mean_bias for r in self.gw_results])
    
    @property
    def mean_baseline_mae(self) -> float:
        return np.mean([r.baseline_mae for r in self.gw_results])
    
    @property
    def mean_improvement_vs_baseline(self) -> float:
        """Mean MAE improvement vs baseline (positive = model better)."""
        return -np.mean([r.mae_vs_baseline for r in self.gw_results])
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-GW results to DataFrame."""
        return pd.DataFrame([
            {
                "gw": r.gw,
                "n_players": r.n_players,
                "mae": r.mae,
                "median_ae": r.median_ae,
                "rmse": r.rmse,
                "spearman": r.spearman_corr,
                "top5_hit": r.top5_hit_rate,
                "top10_hit": r.top10_hit_rate,
                "regret_5": r.regret_5,
                "regret_10": r.regret_10,
                "bias": r.mean_bias,
                "baseline_mae": r.baseline_mae,
                "mae_vs_baseline": r.mae_vs_baseline,
            }
            for r in self.gw_results
        ])
    
    def get_regret_diagnosis_summary(self) -> Dict:
        """Aggregate regret diagnosis across all GWs."""
        diagnoses = [r.regret_diagnosis for r in self.gw_results if r.regret_diagnosis]
        if not diagnoses:
            return {}
        
        return {
            "avg_false_positives": np.mean([d.false_positive_count for d in diagnoses]),
            "avg_false_positive_regret": np.mean([d.false_positive_regret for d in diagnoses]),
            "avg_missed_from_6_10": np.mean([d.missed_from_rank_6_10 for d in diagnoses]),
            "avg_missed_from_11_plus": np.mean([d.missed_from_rank_11_plus for d in diagnoses]),
            "avg_top_scorer_predicted_rank": np.mean([d.top_scorer_predicted_rank for d in diagnoses]),
            "avg_top_scorer_points": np.mean([d.top_scorer_points for d in diagnoses]),
            "avg_our_top_pick_actual_rank": np.mean([d.our_top_pick_actual_rank for d in diagnoses]),
            "avg_our_top_pick_points": np.mean([d.our_top_pick_points for d in diagnoses]),
            # Fraction of GWs where we correctly identified the top scorer
            "top_scorer_identified_rate": np.mean([
                1 if d.top_scorer_predicted_rank == 1 else 0 for d in diagnoses
            ]),
        }
    
    def print_summary(self) -> None:
        """Print formatted backtest summary."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Test GWs: {self.test_gws[0]}-{self.test_gws[-1]} ({len(self.test_gws)} weeks)")
        print(f"Total predictions: {self.total_predictions:,}")
        print(f"Min training GWs: {self.min_train_gws}")
        
        print("\n--- Accuracy ---")
        print(f"MAE:       {self.mean_mae:.3f} ± {self.std_mae:.3f}")
        print(f"RMSE:      {self.mean_rmse:.3f}")
        
        print("\n--- Ranking Quality ---")
        print(f"Spearman:  {self.mean_spearman:.3f}")
        print(f"Top-5 hit: {self.mean_top5_hit_rate:.1%}")
        print(f"Top-10 hit:{self.mean_top10_hit_rate:.1%}")
        
        print("\n--- Decision Cost ---")
        print(f"Regret@5:  {self.mean_regret_5:.1f} pts/GW")
        print(f"Regret@10: {self.mean_regret_10:.1f} pts/GW")
        
        # Regret diagnosis
        diag = self.get_regret_diagnosis_summary()
        if diag:
            print("\n--- Regret Diagnosis (Top-5) ---")
            print(f"False positives/GW:     {diag['avg_false_positives']:.1f} of 5 picks were wrong")
            print(f"  └─ Regret from FPs:   {diag['avg_false_positive_regret']:.1f} pts/GW")
            print(f"Missed top-5 breakdown:")
            print(f"  └─ Near misses (6-10):{diag['avg_missed_from_6_10']:.1f}/GW (ranked close)")
            print(f"  └─ Big misses (11+):  {diag['avg_missed_from_11_plus']:.1f}/GW (blind spots)")
            print(f"Top scorer analysis:")
            print(f"  └─ Avg pts:           {diag['avg_top_scorer_points']:.1f}")
            print(f"  └─ We ranked them:    #{diag['avg_top_scorer_predicted_rank']:.0f} on avg")
            print(f"  └─ Identified #1:     {diag['top_scorer_identified_rate']:.0%} of GWs")
            print(f"Our #1 pick analysis:")
            print(f"  └─ Actual avg rank:   #{diag['avg_our_top_pick_actual_rank']:.0f}")
            print(f"  └─ Actual avg pts:    {diag['avg_our_top_pick_points']:.1f}")
        
        print("\n--- Bias ---")
        print(f"Mean Bias: {self.mean_bias:+.3f} (positive = over-predict)")
        
        print("\n--- vs Baseline ---")
        print(f"Baseline MAE:    {self.mean_baseline_mae:.3f}")
        print(f"Model MAE:       {self.mean_mae:.3f}")
        print(f"Improvement:     {self.mean_improvement_vs_baseline:+.3f} pts")
        print("=" * 60)


class BacktestRunner:
    """Walk-forward backtesting for FPL predictions.
    
    For each test GW t:
        1. Train model on all data where gw < t
        2. Filter out unavailable players (is_inactive=True)
        3. Generate predictions for gw == t
        4. Compute accuracy, ranking, and decision metrics
        5. Compare against rolling-mean baseline
    
    Example:
        runner = BacktestRunner(min_train_gws=5)
        results = runner.run(df)
        results.print_summary()
        
        # Get per-GW DataFrame
        gw_df = results.to_dataframe()
        
        # For regret analysis, store player predictions
        results = runner.run(df, store_predictions=True)
    """
    
    def __init__(
        self,
        min_train_gws: int = 5,
        gw_column: str = "gw",
        target_column: str = "target_points",
        inactive_column: str = "is_inactive",
        verbose: bool = True,
    ):
        """Initialize backtest runner.
        
        Args:
            min_train_gws: Minimum GWs of training data before first prediction.
            gw_column: Column containing gameweek numbers.
            target_column: Column containing actual points (target).
            inactive_column: Column indicating player unavailability.
            verbose: Print progress during backtest.
        """
        self.min_train_gws = min_train_gws
        self.gw_column = gw_column
        self.target_column = target_column
        self.inactive_column = inactive_column
        self.verbose = verbose
        
        # Fixed hyperparameters (no tuning inside backtest)
        self.lgb_params = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": 0.07,
            "num_leaves": 8,
            "min_data_in_leaf": 150,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": 42,
        }
        self.max_rounds = 100
    
    def _train_model(self, train_df: pd.DataFrame):
        """Train LightGBM on training data."""
        import lightgbm as lgb
        
        X = train_df[FEATURE_COLUMNS].values
        y = train_df[self.target_column].values
        
        lgb_train = lgb.Dataset(X, label=y)
        return lgb.train(
            self.lgb_params,
            lgb_train,
            num_boost_round=self.max_rounds,
        )
    
    def _filter_available(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only available players."""
        if self.inactive_column in df.columns:
            return df[df[self.inactive_column] == False].copy()
        return df.copy()
    
    def _build_player_predictions(
        self,
        test_df: pd.DataFrame,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> List[PlayerPrediction]:
        """Build player-level prediction records for regret analysis."""
        n = len(y_pred)
        
        # Compute ranks (1 = highest)
        pred_ranks = n - np.argsort(np.argsort(y_pred))
        actual_ranks = n - np.argsort(np.argsort(y_true))
        
        predictions = []
        for i, (_, row) in enumerate(test_df.reset_index().iterrows()):
            predictions.append(PlayerPrediction(
                player_id=int(row.get("player_id", 0)),
                player_name=str(row.get("player_name", "")),
                position=str(row.get("position", "")),
                predicted_points=float(y_pred[i]),
                actual_points=float(y_true[i]),
                predicted_rank=int(pred_ranks[i]),
                actual_rank=int(actual_ranks[i]),
                minutes=float(row.get("mins_mean", 0.0)),
                # Diagnostic features
                xg_sum=float(row.get("xg_sum", 0.0)),
                xa_sum=float(row.get("xa_sum", 0.0)),
                threat_mean=float(row.get("threat_mean", 0.0)),
                creativity_mean=float(row.get("creativity_mean", 0.0)),
                ict_mean=float(row.get("ict_mean", 0.0)),
                bonus_sum=float(row.get("bonus_sum", 0.0)),
                start_rate_5=float(row.get("start_rate_5", 0.0)),
                mins_std_5=float(row.get("mins_std_5", 0.0)),
            ))
        
        return predictions
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_baseline: np.ndarray,
    ) -> Dict:
        """Compute all metrics for a single GW."""
        n = len(y_true)
        
        # Core accuracy
        mae = mean_absolute_error(y_true, y_pred)
        median_ae = np.median(np.abs(y_true - y_pred))
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Ranking (handle constant arrays)
        if n >= 3 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            spearman_corr, spearman_pval = spearmanr(y_true, y_pred)
            # Handle NaN from spearmanr
            if np.isnan(spearman_corr):
                spearman_corr, spearman_pval = 0.0, 1.0
        else:
            spearman_corr, spearman_pval = 0.0, 1.0
        
        # Top-K hit rates and regret
        top5_hit, regret_5 = self._compute_topk_metrics(y_true, y_pred, k=5)
        top10_hit, regret_10 = self._compute_topk_metrics(y_true, y_pred, k=10)
        
        # Bias
        mean_bias = np.mean(y_pred - y_true)
        
        # Baseline comparison
        baseline_mae = mean_absolute_error(y_true, y_baseline)
        
        return {
            "mae": mae,
            "median_ae": median_ae,
            "rmse": rmse,
            "spearman_corr": spearman_corr,
            "spearman_pval": spearman_pval,
            "top5_hit_rate": top5_hit,
            "top10_hit_rate": top10_hit,
            "regret_5": regret_5,
            "regret_10": regret_10,
            "regret_diagnosis": self._diagnose_regret(y_true, y_pred, k=5),
            "mean_bias": mean_bias,
            "baseline_mae": baseline_mae,
            "mae_vs_baseline": mae - baseline_mae,
        }
    
    def _diagnose_regret(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 5,
    ) -> RegretDiagnosis:
        """Diagnose sources of regret for top-K predictions.
        
        Breaks down regret into:
        - False positives: We picked them top-K, but they weren't
        - False negatives: They were top-K, but we didn't pick them
        - Near misses (ranked 6-10) vs big misses (ranked 11+)
        """
        n = len(y_true)
        k = min(k, n)
        
        if k == 0:
            return RegretDiagnosis(0, 0.0, 0, 0, 0, 0, 0.0, 0, 0.0)
        
        # Compute ranks (1 = highest, n = lowest)
        pred_ranks = n - np.argsort(np.argsort(y_pred))  # 1-indexed ranks
        actual_ranks = n - np.argsort(np.argsort(y_true))
        
        # Sets of indices
        pred_topk_idx = set(np.where(pred_ranks <= k)[0])
        actual_topk_idx = set(np.where(actual_ranks <= k)[0])
        
        # False positives: in pred_topk but not actual_topk
        false_positive_idx = pred_topk_idx - actual_topk_idx
        false_positive_count = len(false_positive_idx)
        
        # False positive regret: points we got from wrong picks vs what top-K would give
        if false_positive_idx:
            # What we got from false positives
            fp_points = sum(y_true[i] for i in false_positive_idx)
            # What we missed (actual top-K players we didn't pick)
            missed_idx = actual_topk_idx - pred_topk_idx
            missed_points = sum(y_true[i] for i in missed_idx)
            false_positive_regret = missed_points - fp_points
        else:
            false_positive_regret = 0.0
        
        # False negatives: in actual_topk but not pred_topk
        false_negative_idx = actual_topk_idx - pred_topk_idx
        false_negative_count = len(false_negative_idx)
        
        # Categorize misses by how badly we ranked them
        missed_from_rank_6_10 = sum(1 for i in false_negative_idx if 6 <= pred_ranks[i] <= 10)
        missed_from_rank_11_plus = sum(1 for i in false_negative_idx if pred_ranks[i] > 10)
        
        # Top scorer analysis
        top_scorer_idx = np.argmax(y_true)
        top_scorer_predicted_rank = int(pred_ranks[top_scorer_idx])
        top_scorer_points = float(y_true[top_scorer_idx])
        
        # Our top pick analysis
        our_top_pick_idx = np.argmax(y_pred)
        our_top_pick_actual_rank = int(actual_ranks[our_top_pick_idx])
        our_top_pick_points = float(y_true[our_top_pick_idx])
        
        return RegretDiagnosis(
            false_positive_count=false_positive_count,
            false_positive_regret=false_positive_regret,
            false_negative_count=false_negative_count,
            missed_from_rank_6_10=missed_from_rank_6_10,
            missed_from_rank_11_plus=missed_from_rank_11_plus,
            top_scorer_predicted_rank=top_scorer_predicted_rank,
            top_scorer_points=top_scorer_points,
            our_top_pick_actual_rank=our_top_pick_actual_rank,
            our_top_pick_points=our_top_pick_points,
        )
    
    def _compute_topk_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int,
    ) -> tuple:
        """Compute Top-K hit rate and regret.
        
        Returns:
            (hit_rate, regret) where:
            - hit_rate: Proportion of actual top-K in predicted top-K
            - regret: Points missed vs optimal top-K selection
        """
        n = len(y_true)
        k = min(k, n)
        
        if k == 0:
            return 0.0, 0.0
        
        # Indices of predicted top-K (highest predictions)
        pred_topk_idx = np.argsort(y_pred)[-k:]
        
        # Indices of actual top-K (highest actual points)
        actual_topk_idx = np.argsort(y_true)[-k:]
        
        # Hit rate: how many of actual top-K did we pick?
        hits = len(set(pred_topk_idx) & set(actual_topk_idx))
        hit_rate = hits / k
        
        # Regret: optimal top-K points minus our top-K points
        optimal_points = y_true[actual_topk_idx].sum()
        our_points = y_true[pred_topk_idx].sum()
        regret = optimal_points - our_points
        
        return hit_rate, regret
    
    def run(
        self,
        df: pd.DataFrame,
        start_gw: Optional[int] = None,
        end_gw: Optional[int] = None,
        store_predictions: bool = False,
    ) -> WalkForwardSummary:
        """Run walk-forward backtest.
        
        Args:
            df: Full feature DataFrame with all GWs.
            start_gw: First GW to test (default: min_train_gws + 1).
            end_gw: Last GW to test (default: max GW in data).
            store_predictions: Store player-level predictions for regret analysis.
            
        Returns:
            WalkForwardSummary with per-GW and aggregate results.
        """
        all_gws = sorted(df[self.gw_column].unique())
        
        # Determine test range
        if start_gw is None:
            start_gw = all_gws[self.min_train_gws]
        if end_gw is None:
            end_gw = all_gws[-1]
        
        test_gws = [gw for gw in all_gws if start_gw <= gw <= end_gw]
        
        if self.verbose:
            print(f"Running backtest: GW {test_gws[0]}-{test_gws[-1]} ({len(test_gws)} weeks)")
        
        results = []
        total_predictions = 0
        
        for test_gw in test_gws:
            # Split: train on all GWs before test_gw
            train_df = df[df[self.gw_column] < test_gw]
            test_df = df[df[self.gw_column] == test_gw]
            
            # Filter unavailable players BEFORE prediction
            test_df = self._filter_available(test_df)
            
            if len(train_df) < 100 or len(test_df) < 10:
                if self.verbose:
                    print(f"  GW {test_gw}: Skipped (insufficient data)")
                continue
            
            # Train model (fresh model for each GW)
            model = self._train_model(train_df)
            
            # Predict
            X_test = test_df[FEATURE_COLUMNS].values
            y_true = test_df[self.target_column].values
            y_pred = model.predict(X_test)
            
            # Baseline predictions
            y_baseline = predict_baseline(test_df)
            
            # Compute metrics
            metrics = self._compute_metrics(y_true, y_pred, y_baseline)
            
            # Store player-level predictions for regret analysis
            player_preds = None
            if store_predictions:
                player_preds = self._build_player_predictions(test_df, y_pred, y_true)
            
            gw_result = GWResult(
                gw=test_gw,
                n_players=len(test_df),
                player_predictions=player_preds,
                **metrics,
            )
            results.append(gw_result)
            total_predictions += len(test_df)
            
            if self.verbose:
                print(f"  GW {test_gw}: MAE={metrics['mae']:.2f}, "
                      f"Spearman={metrics['spearman_corr']:.2f}, "
                      f"Top5={metrics['top5_hit_rate']:.0%}")
        
        return WalkForwardSummary(
            gw_results=results,
            min_train_gws=self.min_train_gws,
            test_gws=test_gws,
            total_predictions=total_predictions,
        )
