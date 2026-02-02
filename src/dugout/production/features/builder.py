"""Feature engineering for FPL point prediction.

Provides the FeatureBuilder class which transforms raw player history
into ML-ready feature vectors. Key responsibilities:
- Decay-weighted performance metrics (per90 points over last 5 games)
- Rolling statistics from last 5 games
- Form indicators (appearances, minutes fraction)
- Start probability estimation

Key Classes:
    FeatureBuilder - Transforms raw data to feature vectors
    FeatureConfig - Configuration for feature computation

Usage:
    from dugout.production.features import FeatureBuilder
    
    builder = FeatureBuilder()
    features = builder.build_features(player_history)
    # Returns dict with all FEATURE_COLUMNS ready for model
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dugout.production.features.definitions import (
    FEATURE_COLUMNS,
    FeatureConfig,
)

if TYPE_CHECKING:
    from dugout.production.data.reader import DataReader

# Re-export for backwards compatibility
DEFAULT_FEATURE_COLUMNS = FEATURE_COLUMNS

__all__ = [
    "FeatureBuilder",
    "FeatureConfig",
    "DEFAULT_FEATURE_COLUMNS",
]


class FeatureBuilder:
    """Builds features for FPL point prediction models."""
    
    # Decay weights for last 5 games: [most_recent, 2nd_recent, 3rd, 4th, oldest]
    # Heavily recency-biased: last week gets 42%, 5 weeks ago nearly ignored (2%)
    # Example: if predicting GW24, weights are [GW23, GW22, GW21, GW20, GW19]
    DECAY_WEIGHTS = np.array([0.4169, 0.2918, 0.2043, 0.0715, 0.0155])
    
    def __init__(
        self, 
        config: Optional[FeatureConfig] = None,
        reader: Optional["DataReader"] = None,
    ) -> None:
        self.config = config or FeatureConfig()
        self.reader = reader
    
    def build_for_player(
        self,
        player_history: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Build features from a single player's gameweek history.
        
        Args:
            player_history: DataFrame of one player's past gameweeks (sorted by gw/round).
                Must be filtered to a single player before calling.
            
        Returns:
            Feature dictionary ready for model prediction.
        """
        if player_history.empty:
            return {}
        
        # Ensure ascending order by gameweek (oldest first)
        gw_col = 'gw' if 'gw' in player_history.columns else 'round'
        player_history = player_history.sort_values(gw_col).reset_index(drop=True)
        
        # Get last 5 games, reversed so index 0 = most recent (aligns with DECAY_WEIGHTS)
        last5 = player_history.tail(5).iloc[::-1].reset_index(drop=True)
        last_row = player_history.iloc[-1]
        
        # Core computed features
        per90_wmean, per90_wvar = self._compute_per90_features(last5)
        rolling = self._compute_minutes_stats(last5)
        stats = self._compute_detailed_stats(last5)
        
        # Player state
        cost_val = self._normalize_cost(last_row.get('now_cost', 0))
        
        features = {
            # Core features
            'per90_wmean': per90_wmean,
            'per90_wvar': per90_wvar,
            'mins_mean': rolling['mins_mean'],
            'appearances': rolling['appearances'],
            'now_cost': cost_val,
            
            # Activity features
            'games_since_first': len(player_history),
            
            # Detailed stats
            **stats,
        }
        
        # Interaction features - products that capture combined effects:
        #   ict_per90_x_mins: Scales ICT rate by playing time (nailed starter with high ICT)
        #   xg_per90_x_apps: Rewards consistent starters with high xG rate
        features['ict_per90_x_mins'] = features['ict_per90'] * features['mins_mean'] / 90 if features['mins_mean'] > 0 else 0
        features['xg_per90_x_apps'] = features['xg_per90'] * features['appearances']
        
        return features
    
    def _compute_per90_features(self, last5: pd.DataFrame) -> Tuple[float, float]:
        """Compute decay-weighted per90 mean and variance from last 5 games.
        
        Features computed:
            - per90_wmean: Weighted average of points-per-90 (recent games weighted higher)
            - per90_wvar: Weighted variance of per90 (measures consistency)
        
        Per90 normalizes points to 90 minutes, so a player scoring 6 pts in 60 mins
        gets per90 = 9.0. Decay weights prioritize recent form (42% to last GW).
        """
        per90_vals = []
        mins_vals = []
        for i in range(len(last5)):
            pts = last5.iloc[i].get('total_points', 0)
            mins = last5.iloc[i].get('minutes', 0)
            per90 = (pts / mins * 90) if mins > 0 else 0
            per90_vals.append(per90)
            mins_vals.append(mins)
        
        # Pad to 5
        while len(per90_vals) < 5:
            per90_vals.append(0)
            mins_vals.append(0)
        
        return self.compute_weighted_per90(per90_vals, mins_vals)
    
    def _compute_minutes_stats(self, last5: pd.DataFrame) -> Dict[str, float]:
        """Compute minutes-related statistics from last 5 games.
        
        Features computed:
            - mins_mean: Average minutes played per game
            - appearances: Count of games with any minutes (> 0)
        """
        if len(last5) == 0:
            return {'mins_mean': 0, 'appearances': 0}
        
        minutes = last5['minutes']
        
        return {
            'mins_mean': minutes.mean(),
            'appearances': int((minutes > 0).sum()),
        }
    
    def _compute_detailed_stats(self, last5: pd.DataFrame) -> Dict[str, float]:
        """Compute detailed performance stats from last 5 games.
        
        All features are per90 (normalized by playing time):
            - goals_per90: Goals per 90 minutes
            - assists_per90: Assists per 90 minutes
            - bonus_per90: Bonus points per 90 minutes
            - bps_per90: BPS per 90 minutes
            - ict_per90: ICT index per 90 minutes (influence + creativity + threat)
            - xg_per90: Expected goals per 90 minutes
            - xa_per90: Expected assists per 90 minutes
        """
        # Total minutes for per90 calculations
        total_mins = last5['minutes'].sum() if 'minutes' in last5.columns else 0
        
        # Handle xG/xA column name variants
        xg_total = self._safe_sum(last5, 'xG') or self._safe_sum(last5, 'expected_goals')
        xa_total = self._safe_sum(last5, 'xA') or self._safe_sum(last5, 'expected_assists')
        
        # Per90 calculation: (total / minutes) * 90
        def to_per90(total: float) -> float:
            return (total / total_mins * 90) if total_mins > 0 else 0.0
        
        return {
            'goals_per90': to_per90(self._safe_sum(last5, 'goals_scored')),
            'assists_per90': to_per90(self._safe_sum(last5, 'assists')),
            'bonus_per90': to_per90(self._safe_sum(last5, 'bonus')),
            'bps_per90': to_per90(self._safe_sum(last5, 'bps')),
            'ict_per90': to_per90(self._safe_sum(last5, 'ict_index')),
            'xg_per90': to_per90(xg_total),
            'xa_per90': to_per90(xa_total),
        }
    
    def _normalize_cost(self, now_cost: Any) -> float:
        """Convert cost from tenths to millions if needed.
        
        Features computed:
            - now_cost: Player price in millions (e.g., 10.5 for £10.5m)
        
        FPL API stores cost in tenths (105 = £10.5m). This normalizes to millions
        by dividing by 10 if value > 15 (no player costs more than £15m).
        """
        cost_val = self._safe_float(now_cost)
        if cost_val > 15:  # Clearly in tenths format (max player ~£15m = 150)
            cost_val = cost_val / 10.0
        return cost_val
    
    def compute_weighted_per90(
        self, 
        per90_values: Sequence[float], 
        minutes_values: Sequence[float]
    ) -> Tuple[float, float]:
        """Compute decay-weighted per90 mean and variance.
        
        Args:
            per90_values: Exactly 5 per90 values (most recent first).
            minutes_values: Exactly 5 minutes values (most recent first).
        
        Returns:
            Tuple of (weighted_mean, weighted_variance)
        
        Algorithm:
            1. Mask out GWs with 0 minutes (player didn't play)
            2. Apply DECAY_WEIGHTS to remaining GWs
            3. Renormalize weights to sum to 1.0
            4. Compute weighted mean: sum(w_i * x_i)
            5. Compute weighted variance: sum(w_i * (x_i - mean)^2)
        
        Weights give 42% to most recent, 2% to oldest. Renormalization ensures
        players who missed games are compared fairly to those who played all 5.
        """
        per90 = np.array(per90_values, dtype=float)
        mins = np.array(minutes_values, dtype=float)
        weights = self.DECAY_WEIGHTS
        
        # Guard: player had 0 minutes in all last 5 GWs (e.g., injured stretch)
        # Upstream filters require total mins > 0, but not recent mins
        mask = mins > 0
        if mask.sum() == 0:
            return 0.0, 0.0
        
        used_weights = weights * mask.astype(float)
        weight_sum = used_weights.sum()
        
        if weight_sum == 0:
            return 0.0, 0.0
        
        used_weights = used_weights / weight_sum
        w_mean = float((used_weights * per90).sum())
        w_var = float((used_weights * (per90 - w_mean) ** 2).sum())
        
        return w_mean, w_var
    
    def build_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Build feature dictionary from a single player row.
        
        DEPRECATED: This method expects old CSV column names (per90_last_1..5,
        goals_last3, etc.) which no longer match current feature definitions.
        Will be updated when training data is regenerated with new schema.
        
        For live predictions, use build_for_player() with raw player history.
        """
        # Extract per90 and minutes arrays
        per90_cols = [f"per90_last_{i}" for i in range(1, 6)]
        mins_cols = [f"mins_last_{i}" for i in range(1, 6)]
        
        per90_vals = [self._safe_float(row.get(c, 0)) for c in per90_cols]
        mins_vals = [self._safe_float(row.get(c, 0)) for c in mins_cols]
        
        per90_wmean, per90_wvar = self.compute_weighted_per90(per90_vals, mins_vals)
        
        features = {
            "per90_wmean": per90_wmean,
            "per90_wvar": per90_wvar,
            "mins_mean": self._safe_float(row.get("mins_mean", 0)),
            "appearances": self._safe_float(row.get("appearances", 0)),
            "now_cost": self._safe_float(row.get("now_cost", 0)),
            "games_since_first": self._safe_float(row.get("games_since_first", 10)),
        }
        
        # Extended features (all per90 from last5)
        extended_cols = [
            "goals_per90", "assists_per90", "bonus_per90", "bps_per90",
            "ict_per90", "xg_per90", "xa_per90",
        ]
        for col in extended_cols:
            features[col] = self._safe_float(row.get(col, 0))
        
        # Interaction features
        ict_per90 = features.get("ict_per90", 0)
        mins_mean = features["mins_mean"]
        xg_per90 = features["xg_per90"]
        apps = features["appearances"]
        
        features["ict_per90_x_mins"] = ict_per90 * mins_mean / 90.0 if mins_mean > 0 else 0
        features["xg_per90_x_apps"] = xg_per90 * apps
        
        return features
    
    def build_training_set(self, raw_df: pd.DataFrame, min_history: int = 5) -> pd.DataFrame:
        """Build training features from raw gameweek data.
        
        For each player-gameweek, builds features from prior gameweeks only
        (no data leakage). Target is the actual points from that gameweek.
        
        Args:
            raw_df: Raw gameweek data with columns:
                player_id, player_name, team_name, team_id, position, gw,
                total_points, minutes, etc.
            min_history: Minimum GWs of history required before creating a row.
                Default 5 ensures full rolling window for features.
                
        Returns:
            DataFrame with metadata, total_points, and feature columns.
        """
        rows = []
        
        if raw_df.empty:
            return pd.DataFrame(rows)
        
        for _, pdf in raw_df.groupby("player_id"):
            pdf = pdf.sort_values("gw").reset_index(drop=True)
            
            for i in range(min_history, len(pdf)):
                target = pdf.iloc[i]
                features = self.build_for_player(pdf.iloc[:i])
                # is_home_next comes from target GW (the one we're predicting)
                features['is_home_next'] = int(target.get('is_home', 0))
                rows.append({
                    **target[["player_id", "player_name", "team_name", "team_id", "position", "gw", "total_points", "minutes"]],
                    **features,
                })
        
        return pd.DataFrame(rows)
    
    def build_for_prediction(
        self,
        raw_df: pd.DataFrame,
        fixture_map: Dict[int, bool],
        min_history: int = 5,
    ) -> pd.DataFrame:
        """Build features for live prediction (no target row).
        
        Unlike build_training_set, this builds features from the latest
        available data and uses fixture_map for is_home_next.
        
        Args:
            raw_df: Raw gameweek data (all history).
            fixture_map: Dict mapping team_id -> is_home (True/False) for target GW.
            min_history: Minimum GWs of history required.
                
        Returns:
            DataFrame with one row per player, ready for prediction.
        """
        rows = []
        
        if raw_df.empty:
            return pd.DataFrame(rows)
        
        for _, pdf in raw_df.groupby("player_id"):
            pdf = pdf.sort_values("gw").reset_index(drop=True)
            
            if len(pdf) < min_history:
                continue
            
            last_row = pdf.iloc[-1]
            features = self.build_for_player(pdf)
            # is_home_next from fixture_map (target GW fixtures)
            team_id = last_row.get('team_id', 0)
            features['is_home_next'] = int(fixture_map.get(team_id, False))
            
            rows.append({
                **last_row[["player_id", "player_name", "team_name", "team_id", "position", "gw", "total_points", "minutes"]],
                **features,
            })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float, returning default if invalid."""
        try:
            return float(value) if value is not None and value != "" else default
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _safe_sum(df: pd.DataFrame, col: str) -> float:
        """Safely sum a column, returning 0 if column doesn't exist."""
        return df[col].sum() if col in df.columns else 0.0
    
    @staticmethod
    def _safe_mean(df: pd.DataFrame, col: str) -> float:
        """Safely compute mean of a column, returning 0 if column doesn't exist."""
        return df[col].mean() if col in df.columns else 0.0
