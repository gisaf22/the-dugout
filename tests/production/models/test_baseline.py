"""Tests for baseline predictor.

Validates the simple heuristic baseline that serves as a comparison point
for ML model performance.
"""

import numpy as np
import pandas as pd
import pytest

from dugout.production.models.baseline import predict_baseline


class TestBaselinePredictor:
    """Test suite for predict_baseline function."""

    def test_high_minutes_player(self):
        """Player with high minutes (>70) should expect 90 mins."""
        df = pd.DataFrame({
            'per90_wmean': [5.0],
            'mins_mean': [80.0]
        })
        pred = predict_baseline(df)
        expected = 5.0 * (90.0 / 90.0)  # 5.0
        assert np.isclose(pred[0], expected)

    def test_low_minutes_player(self):
        """Player with low minutes (<70) should use their average."""
        df = pd.DataFrame({
            'per90_wmean': [3.0],
            'mins_mean': [60.0]
        })
        pred = predict_baseline(df)
        expected = 3.0 * (60.0 / 90.0)  # 2.0
        assert np.isclose(pred[0], expected)

    def test_boundary_minutes(self):
        """Test boundary case at 71 mins (should expect 90 since >70)."""
        df = pd.DataFrame({
            'per90_wmean': [4.0],
            'mins_mean': [71.0]
        })
        pred = predict_baseline(df)
        expected = 4.0 * (90.0 / 90.0)  # 4.0
        assert np.isclose(pred[0], expected)

    def test_just_below_boundary(self):
        """Test just below 70 mins (should use average)."""
        df = pd.DataFrame({
            'per90_wmean': [4.0],
            'mins_mean': [69.9]
        })
        pred = predict_baseline(df)
        expected = 4.0 * (69.9 / 90.0)
        assert np.isclose(pred[0], expected)

    def test_multiple_players(self):
        """Test with multiple players of different minutes."""
        df = pd.DataFrame({
            'per90_wmean': [5.0, 3.0, 4.0],
            'mins_mean': [80.0, 60.0, 90.0]
        })
        pred = predict_baseline(df)
        expected = np.array([
            5.0 * (90.0 / 90.0),  # High mins → 90
            3.0 * (60.0 / 90.0),  # Low mins → 60
            4.0 * (90.0 / 90.0)   # Very high mins → 90
        ])
        assert np.allclose(pred, expected)

    def test_missing_per90_column(self):
        """Missing per90_wmean column should default to 0."""
        df = pd.DataFrame({
            'mins_mean': [80.0, 60.0]
        })
        pred = predict_baseline(df)
        assert np.allclose(pred, [0.0, 0.0])

    def test_missing_mins_column(self):
        """Missing mins_mean column should default to 0."""
        df = pd.DataFrame({
            'per90_wmean': [5.0, 3.0]
        })
        pred = predict_baseline(df)
        assert np.allclose(pred, [0.0, 0.0])

    def test_both_columns_missing(self):
        """Missing both columns should return all zeros."""
        df = pd.DataFrame({
            'other_col': [1, 2, 3]
        })
        pred = predict_baseline(df)
        assert np.allclose(pred, [0.0, 0.0, 0.0])

    def test_nan_in_per90(self):
        """NaN in per90_wmean should be treated as 0."""
        df = pd.DataFrame({
            'per90_wmean': [5.0, np.nan, 3.0],
            'mins_mean': [80.0, 75.0, 65.0]
        })
        pred = predict_baseline(df)
        expected = np.array([
            5.0 * (90.0 / 90.0),  # 5.0
            0.0,                   # NaN → 0
            3.0 * (65.0 / 90.0)   # 2.167
        ])
        assert np.allclose(pred, expected)

    def test_nan_in_mins(self):
        """NaN in mins_mean should be treated as 0."""
        df = pd.DataFrame({
            'per90_wmean': [5.0, 3.0],
            'mins_mean': [80.0, np.nan]
        })
        pred = predict_baseline(df)
        expected = np.array([
            5.0 * (90.0 / 90.0),  # 5.0
            3.0 * (0.0 / 90.0)    # 0.0
        ])
        assert np.allclose(pred, expected)

    def test_zero_per90(self):
        """Player with 0 per90 should predict 0."""
        df = pd.DataFrame({
            'per90_wmean': [0.0],
            'mins_mean': [90.0]
        })
        pred = predict_baseline(df)
        assert np.isclose(pred[0], 0.0)

    def test_zero_minutes(self):
        """Player with 0 minutes should predict 0."""
        df = pd.DataFrame({
            'per90_wmean': [5.0],
            'mins_mean': [0.0]
        })
        pred = predict_baseline(df)
        assert np.isclose(pred[0], 0.0)

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty array."""
        df = pd.DataFrame({
            'per90_wmean': [],
            'mins_mean': []
        })
        pred = predict_baseline(df)
        assert len(pred) == 0

    def test_return_type(self):
        """Should return numpy array."""
        df = pd.DataFrame({
            'per90_wmean': [5.0],
            'mins_mean': [80.0]
        })
        pred = predict_baseline(df)
        assert isinstance(pred, np.ndarray)

    def test_real_world_scenario(self):
        """Test with realistic player scores."""
        df = pd.DataFrame({
            'per90_wmean': [3.5, 2.1, 4.8, 1.3, 5.2],
            'mins_mean': [88.0, 45.0, 72.0, 15.0, 90.0]
        })
        pred = predict_baseline(df)
        expected = np.array([
            3.5 * (90.0 / 90.0),    # Regular starter
            2.1 * (45.0 / 90.0),    # Bench player
            4.8 * (90.0 / 90.0),    # Solid contributor
            1.3 * (15.0 / 90.0),    # Rare appearance
            5.2 * (90.0 / 90.0)     # Star player
        ])
        assert np.allclose(pred, expected)
