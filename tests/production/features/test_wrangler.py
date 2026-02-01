"""Tests for Wrangler data cleaning and formatting utilities."""

import pandas as pd
import pytest
from unittest.mock import Mock, patch

from dugout.production.features.wrangler import Wrangler


@pytest.fixture
def sample_gw_data():
    """Sample raw GW data (before cleaning)."""
    return pd.DataFrame({
        "element_id": [1, 1, 2, 2, 3],
        "round": [1, 2, 1, 2, 1],
        "element_type": [1, 1, 2, 2, 3],  # 1=GKP, 2=DEF, 3=MID
        "now_cost": [50, 50, 60, 60, 75],  # in tenths (£5.0m, £6.0m, £7.5m)
        "status": ["a", "a", "i", "a", "u"],  # a=available, i=injured, u=unavailable
        "total_points": [3, 5, 0, 2, 8],
    })


@pytest.fixture
def wrangler():
    """Wrangler instance with mocked reader."""
    reader = Mock()
    return Wrangler(reader=reader)


class TestRemoveDuplicates:
    """Tests for _remove_duplicates() method."""
    
    def test_removes_duplicate_player_gameweeks(self, wrangler):
        """Should remove duplicate (element_id, round) pairs."""
        df = pd.DataFrame({
            "element_id": [1, 1, 1, 2],  # element 1 appears 3 times in round 1
            "round": [1, 1, 2, 1],
            "total_points": [3, 5, 6, 2],  # Different points for duplicates
        })
        result = wrangler._remove_duplicates(df)
        # Should keep only first occurrence
        assert len(result) == 3  # One from the duplicates, plus others
        assert (result[(result["element_id"] == 1) & (result["round"] == 1)]["total_points"].values == [3]).all()
    
    def test_keeps_first_occurrence(self, wrangler):
        """Should keep 'first' when removing duplicates."""
        df = pd.DataFrame({
            "element_id": [1, 1],
            "round": [1, 1],
            "total_points": [10, 20],
        })
        result = wrangler._remove_duplicates(df)
        assert len(result) == 1
        assert result["total_points"].iloc[0] == 10  # First value kept
    
    def test_no_duplicates_unchanged(self, sample_gw_data, wrangler):
        """Should return unchanged if no duplicates."""
        result = wrangler._remove_duplicates(sample_gw_data)
        assert len(result) == len(sample_gw_data)


class TestFormatColumns:
    """Tests for _format_columns() method."""
    
    def test_converts_element_type_to_position(self, sample_gw_data, wrangler):
        """Should map element_type codes to position strings."""
        result = wrangler._format_columns(sample_gw_data)
        assert "position" in result.columns
        assert result.iloc[0]["position"] == "GKP"  # element_type=1
        assert result.iloc[2]["position"] == "DEF"  # element_type=2
        assert result.iloc[4]["position"] == "MID"  # element_type=3
    
    def test_handles_missing_columns(self, wrangler):
        """Should handle missing element_type gracefully."""
        df = pd.DataFrame({"element_id": [1], "round": [1]})
        result = wrangler._format_columns(df)
        # Should not crash, return dataframe with same rows
        assert len(result) == 1
    
    def test_does_not_mutate_input(self, sample_gw_data, wrangler):
        """Should not modify input DataFrame."""
        original_copy = sample_gw_data.copy()
        wrangler._format_columns(sample_gw_data)
        pd.testing.assert_frame_equal(sample_gw_data, original_copy)


class TestGetPredictionReadyData:
    """Tests for get_prediction_ready_data() orchestrator method."""
    
    def test_orchestrates_full_pipeline(self, sample_gw_data, wrangler):
        """Should apply cleaning steps without status filtering."""
        wrangler.reader.get_all_gw_data.return_value = sample_gw_data

        result = wrangler.get_prediction_ready_data()

        # Should have:
        # - Kept all rows (no status filtering, preserve training data)
        # - No duplicates
        # - Formatted columns (position added; cost left to FeatureBuilder)
        assert len(result) == 5  # All rows kept
        assert "position" in result.columns
        assert "now_cost" in result.columns  # Cost stays as now_cost for FeatureBuilder
        # Should preserve all status values for training
        assert "a" in result["status"].values
        assert "u" in result["status"].values  # Unavailable kept for training


class TestNormalizePosition:
    """Tests for normalize_position() static method."""
    
    @pytest.mark.parametrize("input_pos,expected", [
        ("GK", "GKP"),
        ("gk", "GKP"),
        ("goalkeeper", "GKP"),
        ("DEF", "DEF"),
        ("defender", "DEF"),
        ("MID", "MID"),
        ("midfielder", "MID"),
        ("FWD", "FWD"),
        ("forward", "FWD"),
        ("attacker", "FWD"),
        ("ATT", "FWD"),
        ("UNKNOWN", "UNKNOWN"),
    ])
    def test_normalizes_position_formats(self, input_pos, expected):
        """Should convert various position formats to standard codes."""
        result = Wrangler.normalize_position(input_pos)
        assert result == expected
    
    def test_handles_whitespace(self):
        """Should strip whitespace."""
        assert Wrangler.normalize_position("  GK  ") == "GKP"
        assert Wrangler.normalize_position("\tDEF\n") == "DEF"
