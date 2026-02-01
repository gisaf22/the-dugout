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
    
    # Decay weights for last 5 gameweeks: [GW-1, GW-2, GW-3, GW-4, GW-5]
    # weight = 0.7^n, normalized to sum to 1.0
    # Most recent game gets ~42% weight, oldest ~2%
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
        minutes_risk = self._compute_minutes_risk_features(last5)
        tail_risk = self._compute_tail_risk_features(last5)
        
        # Fixture and player state
        is_home = self._lookup_is_home(last_row)
        cost_val = self._normalize_cost(last_row.get('now_cost', 0))
        status = last_row.get('status', '')
        
        features = {
            # Core features
            'per90_wmean': per90_wmean,
            'per90_wvar': per90_wvar,
            'mins_mean': rolling['mins_mean'],
            'appearances': rolling['appearances'],
            'now_cost': cost_val,
            
            # Fixture features
            'is_home_next': 1 if is_home else 0,
            'is_inactive': 1 if str(status).lower() in {'i', 's', 'u', 'n'} else 0,
            
            # Activity features
            'games_since_first': len(player_history),
            'completed_60_count': rolling['completed_60_count'],
            'minutes_fraction': rolling['minutes_fraction'],
            
            # Detailed stats
            **stats,
            
            # Minutes risk features
            **minutes_risk,
            
            # Tail risk / volatility features
            **tail_risk,
        }
        
        # Interaction features - products that capture combined effects:
        #   threat_x_mins: Scales threat by playing time (90-min player keeps full value)
        #   ict_x_home: Measures home performance (ICT when home, 0 when away)
        #   xg_x_apps: Rewards consistent starters who generate chances
        features['threat_x_mins'] = features['threat_mean'] * features['mins_mean'] / 90 if features['mins_mean'] > 0 else 0
        features['ict_x_home'] = features['ict_mean'] * features['is_home_next']
        features['xg_x_apps'] = features['xg_sum'] * features['appearances']
        
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
            - completed_60_count: Count of games with 60+ minutes (likely starts)
            - minutes_fraction: Total mins / (90 * games) - playing time share
        """
        if len(last5) == 0:
            return {'mins_mean': 0, 'appearances': 0, 'completed_60_count': 0, 'minutes_fraction': 0}
        
        total_possible = 90 * len(last5)
        return {
            'mins_mean': last5['minutes'].mean(),
            'appearances': int((last5['minutes'] > 0).sum()),
            'completed_60_count': int((last5['minutes'] >= 60).sum()),
            'minutes_fraction': last5['minutes'].sum() / total_possible if total_possible > 0 else 0,
        }
    
    def _compute_minutes_risk_features(self, last5: pd.DataFrame) -> Dict[str, float]:
        """Compute minutes risk features from last 5 games.
        
        Features computed:
            - start_rate_5: Proportion of games started (0.0-1.0)
            - mins_std_5: Standard deviation of minutes (volatility)
            - mins_below_60_rate_5: Proportion of games with < 60 mins
        
        These capture rotation/benching risk patterns. A player with
        start_rate_5=0.6 and mins_below_60_rate_5=0.4 is a rotation risk.
        
        Note: These use the SAME last5 window as other features, which is
        already lagged (built from games BEFORE the target GW).
        """
        if len(last5) == 0:
            return {
                'start_rate_5': 0.0,
                'mins_std_5': 0.0,
                'mins_below_60_rate_5': 0.0,
            }
        
        n_games = len(last5)
        minutes = last5['minutes']
        
        # P(mins >= 60): proportion of games with 60+ minutes
        # This is the empirical estimator for "full match" probability
        mins_ge_60_rate = (minutes >= 60).sum() / n_games
        
        # Minutes volatility
        mins_std = minutes.std() if n_games > 1 else 0.0
        
        # Proportion of low-minute games (< 60 mins = likely sub or benched)
        mins_below_60_rate = (minutes < 60).sum() / n_games
        
        return {
            'start_rate_5': float(mins_ge_60_rate),  # P(mins >= 60)
            'mins_std_5': float(mins_std) if not pd.isna(mins_std) else 0.0,
            'mins_below_60_rate_5': float(mins_below_60_rate),
        }
    
    def _compute_tail_risk_features(self, last5: pd.DataFrame) -> Dict[str, float]:
        """Compute tail risk / upside features from last 5 games.
        
        Features computed:
            - haul_rate_5: Proportion of games with 10+ points (right-tail frequency)
        
        A player with haul_rate_5=0.4 hauled in 2 of last 5 GWs. This directly
        encodes ceiling potential rather than variance (which tree models treat
        as risk, not opportunity).
        
        Note: Uses the SAME last5 window as other features, already lagged.
        """
        if len(last5) == 0:
            return {'haul_rate_5': 0.0}
        
        points = last5['total_points']
        n_games = len(last5)
        
        # Haul rate: proportion of games with 10+ points
        haul_count = (points >= 10).sum()
        haul_rate = haul_count / n_games
        
        return {
            'haul_rate_5': float(haul_rate),
        }
    
    def _compute_detailed_stats(self, last5: pd.DataFrame) -> Dict[str, float]:
        """Compute detailed performance stats from last 5 games.
        
        Features computed (sums over last 5):
            - goals_sum: Total goals scored
            - assists_sum: Total assists
            - bonus_sum: Total bonus points
            - xg_sum: Expected goals (xG)
            - xa_sum: Expected assists (xA)
        
        Features computed (means over last 5):
            - bps_mean: Average bonus point system score
            - creativity_mean: Average creativity index
            - threat_mean: Average threat index
            - influence_mean: Average influence index
            - ict_mean: Average ICT index (combined)
        """
        def safe_sum(col: str) -> float:
            return last5[col].sum() if col in last5.columns else 0
        
        def safe_mean(col: str) -> float:
            return last5[col].mean() if col in last5.columns else 0
        
        # Handle xG/xA column name variants
        xg = safe_sum('xG') or safe_sum('expected_goals')
        xa = safe_sum('xA') or safe_sum('expected_assists')
        
        return {
            'goals_sum': safe_sum('goals_scored'),
            'assists_sum': safe_sum('assists'),
            'bonus_sum': safe_sum('bonus'),
            'bps_mean': safe_mean('bps'),
            'creativity_mean': safe_mean('creativity'),
            'threat_mean': safe_mean('threat'),
            'influence_mean': safe_mean('influence'),
            'ict_mean': safe_mean('ict_index'),
            'xg_sum': xg,
            'xa_sum': xa,
        }
    
    def _lookup_is_home(self, last_row: pd.Series) -> int:
        """Look up is_home from fixtures table for next GW.
        
        Features computed:
            - is_home_next: 1 if team plays at home next GW, 0 if away
        
        Uses DataReader to query fixtures table. Returns 0 if reader unavailable
        or fixture not found (blank GW).
        """
        if self.reader is None:
            return 0
        
        team_id = last_row.get('team_id') or last_row.get('team')
        last_gw = last_row.get('round') or last_row.get('gw') or last_row.get('GW')
        
        if team_id is None or last_gw is None:
            return 0
        
        try:
            return self.reader.is_home_fixture(int(team_id), int(last_gw) + 1)
        except (KeyError, TypeError):
            return 0
    
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
            "total_points_season": self._safe_float(row.get("total_points_season", 0)),
            "is_home_next": 1 if row.get("is_home_next") else 0,
            "is_inactive": 1 if row.get("status", "").lower() in {"i", "s", "u", "n"} else 0,
            "completed_60_count": self._safe_float(row.get("completed_60_count", 0)),
            "minutes_fraction": self._safe_float(row.get("minutes_fraction", 0)),
        }
        
        # Extended features (rolling stats from last5)
        extended_cols = [
            "goals_sum", "assists_sum", "bonus_sum", "bps_mean",
            "creativity_mean", "threat_mean", "influence_mean", "ict_mean",
            "xg_sum", "xa_sum",
        ]
        for col in extended_cols:
            features[col] = self._safe_float(row.get(col, 0))
        
        # Minutes risk features
        features["start_rate_5"] = self._safe_float(row.get("start_rate_5", 1.0))
        features["mins_std_5"] = self._safe_float(row.get("mins_std_5", 0.0))
        features["mins_below_60_rate_5"] = self._safe_float(row.get("mins_below_60_rate_5", 0.0))
        
        # Tail risk / upside features
        features["haul_rate_5"] = self._safe_float(row.get("haul_rate_5", 0.0))
        
        # Games since first appearance
        features["games_since_first"] = self._safe_float(row.get("games_since_first", 10))
        
        # Interaction features
        threat_mean = features.get("threat_mean", 0)
        mins_mean = features["mins_mean"]
        ict_mean = features["ict_mean"]
        xg_sum = features["xg_sum"]
        apps = features["appearances"]
        is_home = features["is_home_next"]
        
        features["threat_x_mins"] = threat_mean * mins_mean / 90.0 if mins_mean > 0 else 0
        features["ict_x_home"] = ict_mean * is_home
        features["xg_x_apps"] = xg_sum * apps if apps > 0 else 0
        
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
                rows.append({
                    **target[["player_id", "player_name", "team_name", "team_id", "position", "gw", "total_points", "minutes"]],
                    **self.build_for_player(pdf.iloc[:i]),
                })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float."""
        if value is None or value == "":
            return default
        try:
            result = float(value)
            if np.isnan(result):
                return default
            return result
        except (ValueError, TypeError):
            return default
