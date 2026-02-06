"""Baseline predictors for model comparison.

Simple heuristic predictors that serve as benchmarks for ML models.
If your model can't beat these, something is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_baseline(df: pd.DataFrame) -> np.ndarray:
    """Naive baseline: per90_wmean × (expected_mins / 90).
    
    Simple predictor for model comparison. Assumes players with avg > 70 mins
    will play 90, otherwise uses their average.
    
    Args:
        df: DataFrame with 'per90_wmean' and 'mins_mean' columns.
        
    Returns:
        Array of predicted points per row.
    """
    per90 = df.get("per90_wmean", pd.Series(0.0, index=df.index)).fillna(0.0)
    mins = df.get("mins_mean", pd.Series(0.0, index=df.index)).fillna(0.0)
    
    per90_arr = np.asarray(per90, dtype=float)
    mins_arr = np.asarray(mins, dtype=float)
    
    # Expect 90 mins if averaging > 70, otherwise use their average
    exp_mins = np.where(mins_arr > 70.0, 90.0, mins_arr)
    
    return per90_arr * (exp_mins / 90.0)
