"""Tests for FeatureBuilder."""

import pandas as pd
import pytest

from dugout.production.features import FeatureBuilder


@pytest.fixture
def sample_raw_df():
    """Create sample raw gameweek data for one player."""
    return pd.DataFrame({
        "player_id": [1] * 10,
        "player_name": ["Salah"] * 10,
        "team_name": ["Liverpool"] * 10,
        "team_id": [1] * 10,
        "position": ["MID"] * 10,
        "gw": list(range(1, 11)),
        "total_points": [gw * 10 for gw in range(1, 11)],  # 10, 20, ..., 100
        "minutes": [90] * 10,
        "goals_scored": [0] * 10,
        "assists": [0] * 10,
        "bonus": [0] * 10,
        "bps": [0] * 10,
        "influence": [0] * 10,
        "creativity": [0] * 10,
        "threat": [0] * 10,
        "ict_index": [0] * 10,
        "expected_goals": [0] * 10,
        "expected_assists": [0] * 10,
        "now_cost": [100] * 10,
        "status": ["a"] * 10,
    })


class TestBuildTrainingSet:
    """Tests for build_training_set method."""
    
    def test_earliest_gw_is_6_with_default_min_history(self, sample_raw_df):
        """With min_history=5, first output row should be GW 6."""
        builder = FeatureBuilder()
        result = builder.build_training_set(sample_raw_df)
        
        assert result["gw"].min() == 6
    
    def test_earliest_gw_is_2_with_min_history_1(self, sample_raw_df):
        """With min_history=1, first output row should be GW 2."""
        builder = FeatureBuilder()
        result = builder.build_training_set(sample_raw_df, min_history=1)
        
        assert result["gw"].min() == 2
    
    def test_no_data_leakage(self, sample_raw_df):
        """Features should only use prior GWs, not target GW."""
        builder = FeatureBuilder()
        result = builder.build_training_set(sample_raw_df)
        
        # GW 6 row: features from GWs 1-5 (5 games of history)
        gw6 = result[result["gw"] == 6].iloc[0]
        assert gw6["games_since_first"] == 5  # 5 prior GWs
        assert gw6["total_points"] == 60  # GW 6 actual points (target, not feature)
        
        # GW 10 row: features from GWs 1-9 (9 games of history)
        gw10 = result[result["gw"] == 10].iloc[0]
        assert gw10["games_since_first"] == 9  # 9 prior GWs
        assert gw10["total_points"] == 100  # GW 10 actual points
    
    def test_output_has_metadata_columns(self, sample_raw_df):
        """Result should include player metadata."""
        builder = FeatureBuilder()
        result = builder.build_training_set(sample_raw_df)
        
        expected_cols = ["player_id", "player_name", "team_name", "team_id", "position", "gw", "total_points"]
        for col in expected_cols:
            assert col in result.columns
    
    def test_output_has_feature_columns(self, sample_raw_df):
        """Result should include computed features."""
        builder = FeatureBuilder()
        result = builder.build_training_set(sample_raw_df)
        
        feature_cols = ["per90_wmean", "mins_mean", "games_since_first", "appearances"]
        for col in feature_cols:
            assert col in result.columns
    
    def test_empty_df_returns_empty(self):
        """Empty input should return empty DataFrame."""
        builder = FeatureBuilder()
        result = builder.build_training_set(pd.DataFrame())
        
        assert len(result) == 0
    
    def test_player_with_insufficient_history_excluded(self):
        """Players with < min_history+1 GWs should be excluded."""
        df = pd.DataFrame({
            "player_id": [1, 1, 1],  # Only 3 GWs
            "player_name": ["Test"] * 3,
            "team_name": ["Team"] * 3,
            "team_id": [1] * 3,
            "position": ["MID"] * 3,
            "gw": [1, 2, 3],
            "total_points": [5, 5, 5],
            "minutes": [90, 90, 90],
        })
        
        builder = FeatureBuilder()
        result = builder.build_training_set(df, min_history=5)
        
        assert len(result) == 0


class TestMinutesRiskFeatures:
    """Tests for minutes risk feature computation."""
    
    def test_start_rate_all_starts(self):
        """Player who starts every game has start_rate_5 = 1.0."""
        df = pd.DataFrame({
            "player_id": [1] * 6,
            "player_name": ["Starter"] * 6,
            "team_name": ["Team"] * 6,
            "team_id": [1] * 6,
            "position": ["MID"] * 6,
            "gw": [1, 2, 3, 4, 5, 6],
            "total_points": [5] * 6,
            "minutes": [90, 90, 90, 90, 90, 90],
            "starts": [1, 1, 1, 1, 1, 1],
        })
        
        builder = FeatureBuilder()
        result = builder.build_training_set(df, min_history=5)
        
        assert len(result) == 1
        assert result.iloc[0]["start_rate_5"] == 1.0
        assert result.iloc[0]["mins_below_60_rate_5"] == 0.0
    
    def test_rotation_player_features(self):
        """Player rotated frequently should have low start_rate_5."""
        df = pd.DataFrame({
            "player_id": [1] * 6,
            "player_name": ["Rotation"] * 6,
            "team_name": ["Team"] * 6,
            "team_id": [1] * 6,
            "position": ["MID"] * 6,
            "gw": [1, 2, 3, 4, 5, 6],
            "total_points": [5] * 6,
            "minutes": [90, 20, 90, 15, 90, 25],  # Alternating starter/sub
            "starts": [1, 0, 1, 0, 1, 0],  # 3/5 starts in history for GW 6
        })
        
        builder = FeatureBuilder()
        result = builder.build_training_set(df, min_history=5)
        
        # For GW 6, history is GWs 1-5: starts = [1,0,1,0,1] = 3/5
        assert result.iloc[0]["start_rate_5"] == 0.6
        # GWs 1-5 minutes: [90,20,90,15,90] → 2/5 below 60
        assert result.iloc[0]["mins_below_60_rate_5"] == 0.4
    
    def test_mins_std_consistent_vs_volatile(self):
        """Volatile minutes should have higher mins_std_5."""
        # Consistent player: always 90 mins
        df_consistent = pd.DataFrame({
            "player_id": [1] * 6,
            "player_name": ["Consistent"] * 6,
            "team_name": ["Team"] * 6,
            "team_id": [1] * 6,
            "position": ["MID"] * 6,
            "gw": [1, 2, 3, 4, 5, 6],
            "total_points": [5] * 6,
            "minutes": [90, 90, 90, 90, 90, 90],
        })
        
        # Volatile player: varies wildly
        df_volatile = pd.DataFrame({
            "player_id": [2] * 6,
            "player_name": ["Volatile"] * 6,
            "team_name": ["Team"] * 6,
            "team_id": [2] * 6,
            "position": ["MID"] * 6,
            "gw": [1, 2, 3, 4, 5, 6],
            "total_points": [5] * 6,
            "minutes": [90, 0, 90, 30, 60, 45],
        })
        
        builder = FeatureBuilder()
        result_c = builder.build_training_set(df_consistent, min_history=5)
        result_v = builder.build_training_set(df_volatile, min_history=5)
        
        # Consistent player has 0 std
        assert result_c.iloc[0]["mins_std_5"] == 0.0
        # Volatile player has higher std
        assert result_v.iloc[0]["mins_std_5"] > 30.0
