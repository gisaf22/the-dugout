"""Evaluation metrics for FPL predictions and captain picks.

Provides tools to measure model performance against actual outcomes.
Used for backtesting and model validation.

Key Classes:
    PredictionMetrics - MAE, RMSE, MAPE, R² for predictions
    CaptainMetrics - Captain pick success rates

Key Functions:
    evaluate_predictions() - Compute prediction accuracy metrics
    captain_metrics() - Evaluate captain recommendation performance

Metrics Explained:
    MAE: Average absolute error (lower is better)
    RMSE: Root mean square error (penalizes large errors)
    Captain Success Rate: % of captains that returned 6+ points
    Top Captain %: % of captains that were the best possible pick

Usage:
    from dugout.production.analysis.models import evaluate_predictions, captain_metrics
    
    metrics = evaluate_predictions(predictions_df, actuals_df)
    print(f"MAE: {metrics.mae:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PredictionMetrics:
    """Metrics for point prediction accuracy."""
    
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    mape: float  # Mean Absolute Percentage Error
    r2: float  # R-squared
    n_samples: int
    
    def to_dict(self) -> dict:
        return {
            "mae": round(self.mae, 3),
            "rmse": round(self.rmse, 3),
            "mape": round(self.mape, 3),
            "r2": round(self.r2, 3),
            "n_samples": self.n_samples,
        }
    
    def __repr__(self) -> str:
        return f"MAE: {self.mae:.2f}, RMSE: {self.rmse:.2f}, R²: {self.r2:.3f} (n={self.n_samples})"


@dataclass
class CaptainMetrics:
    """Metrics for captain selection performance."""
    
    total_points: float  # Total points from captain picks (doubled)
    avg_points: float  # Average points per gameweek
    hit_rate: float  # Proportion of correct calls (beat mean)
    haul_rate: float  # Proportion of 10+ point captains
    blank_rate: float  # Proportion of <4 point captains
    vs_optimal: float  # Points difference vs perfect hindsight
    n_picks: int
    
    def to_dict(self) -> dict:
        return {
            "total_points": round(self.total_points, 1),
            "avg_points": round(self.avg_points, 2),
            "hit_rate": f"{self.hit_rate:.1%}",
            "haul_rate": f"{self.haul_rate:.1%}",
            "blank_rate": f"{self.blank_rate:.1%}",
            "vs_optimal": round(self.vs_optimal, 1),
            "n_picks": self.n_picks,
        }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> PredictionMetrics:
    """Calculate prediction metrics.
    
    Args:
        y_true: Actual points
        y_pred: Predicted points
        
    Returns:
        PredictionMetrics with MAE, RMSE, etc.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    n = len(y_true)
    if n == 0:
        return PredictionMetrics(0, 0, 0, 0, 0)
    
    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # MAPE (handle zeros)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_values = np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))
        mape = float(np.mean(mape_values[np.isfinite(mape_values)])) if np.any(np.isfinite(mape_values)) else 0
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
    
    return PredictionMetrics(
        mae=mae,
        rmse=rmse,
        mape=mape,
        r2=r2,
        n_samples=n,
    )


def captain_metrics(
    picks: List[Dict],
    actuals: Dict[int, int],
) -> CaptainMetrics:
    """Calculate captain selection metrics.
    
    Args:
        picks: List of captain pick dicts with player_id, predicted_ev
        actuals: Dict mapping player_id to actual points
        
    Returns:
        CaptainMetrics summarizing performance
    """
    if not picks:
        return CaptainMetrics(0, 0, 0, 0, 0, 0, 0)
    
    points_list = []
    hits = 0
    hauls = 0
    blanks = 0
    
    # Calculate optimal (best possible captain each week)
    optimal_total = 0
    
    for pick in picks:
        player_id = pick["player_id"]
        actual = actuals.get(player_id, 0)
        doubled = actual * 2
        points_list.append(doubled)
        
        # Hit: beat mean of candidates
        mean_points = pick.get("mean_candidate_points", 3)  # Default to 3
        if actual >= mean_points:
            hits += 1
        
        # Haul: 10+ points
        if actual >= 10:
            hauls += 1
        
        # Blank: < 4 points
        if actual < 4:
            blanks += 1
        
        # Optimal (would need all candidates' actuals for true optimal)
        optimal_total += pick.get("optimal_points", doubled)
    
    n = len(picks)
    total = sum(points_list)
    
    return CaptainMetrics(
        total_points=total,
        avg_points=total / n if n > 0 else 0,
        hit_rate=hits / n if n > 0 else 0,
        haul_rate=hauls / n if n > 0 else 0,
        blank_rate=blanks / n if n > 0 else 0,
        vs_optimal=optimal_total - total,
        n_picks=n,
    )


def evaluate_by_position(
    df: pd.DataFrame,
    y_true_col: str = "total_points",
    y_pred_col: str = "predicted",
    position_col: str = "position",
) -> Dict[str, PredictionMetrics]:
    """Evaluate predictions grouped by position.
    
    Args:
        df: DataFrame with predictions and actuals
        y_true_col: Column name for actual points
        y_pred_col: Column name for predictions
        position_col: Column name for player position
        
    Returns:
        Dict mapping position to PredictionMetrics
    """
    results = {}
    
    for pos in df[position_col].unique():
        pos_df = df[df[position_col] == pos]
        metrics = evaluate_predictions(
            pos_df[y_true_col].values,
            pos_df[y_pred_col].values,
        )
        results[pos] = metrics
    
    return results


def evaluate_haul_detection(
    y_true: np.ndarray,
    haul_probs: np.ndarray,
    threshold: float = 0.15,
) -> Dict[str, float]:
    """Evaluate haul detection accuracy.
    
    Args:
        y_true: Actual points
        haul_probs: Predicted probability of 10+ points
        threshold: Probability threshold to call a haul
        
    Returns:
        Dict with precision, recall, F1, etc.
    """
    y_true = np.asarray(y_true).flatten()
    haul_probs = np.asarray(haul_probs).flatten()
    
    # Actual hauls
    actual_haul = y_true >= 10
    
    # Predicted hauls
    predicted_haul = haul_probs >= threshold
    
    # True positives, false positives, etc.
    tp = np.sum(predicted_haul & actual_haul)
    fp = np.sum(predicted_haul & ~actual_haul)
    fn = np.sum(~predicted_haul & actual_haul)
    tn = np.sum(~predicted_haul & ~actual_haul)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "haul_rate": round(np.mean(actual_haul), 3),
    }
