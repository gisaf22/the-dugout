"""Captain selection logic for FPL.

Frozen Decision Rule (Stage 6):
    Captain = argmax(predicted_points)
    
    The player with the highest predicted points is selected as captain.
    
    IMPORTANT: Availability weighting (p_play × expected_points) was tested
    and explicitly REJECTED for single-GW captain decisions. Research showed
    it increases regret compared to pure expected points selection.

Key Classes:
    CaptainPicker - Main captain selection class
    CaptainRecommendation - Structured captain pick with context

Usage:
    from dugout.production.models import CaptainPicker
    
    picker = CaptainPicker(predictions_df)
    pick = picker.get_recommendation()
    print(f"Captain: {pick.player_name} (EV×2: {pick.ev_doubled})")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class CaptainRecommendation:
    """A captain recommendation with context.
    
    Attributes:
        player_id: FPL player ID.
        player_name: Player display name.
        team_name: Team name.
        ev_doubled: Expected value × 2 (captain multiplier).
        p_start: Probability of starting.
    """
    
    player_id: int
    player_name: str
    team_name: str
    ev_doubled: float
    p_start: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team_name": self.team_name,
            "ev_doubled": round(self.ev_doubled, 2),
            "p_start": round(self.p_start, 3),
        }


class CaptainPicker:
    """Selects captain using argmax(predicted_points).
    
    Frozen decision rule: Pick the player with highest predicted_points.
    
    Example:
        >>> picker = CaptainPicker(predictions_df)
        >>> pick = picker.get_recommendation(squad_ids)
        >>> print(f"Captain: {pick.player_name} (EV×2: {pick.ev_doubled})")
    """
    
    def __init__(self, predictions_df: pd.DataFrame) -> None:
        """Initialize captain picker.
        
        Args:
            predictions_df: DataFrame with predictions including
                player_id, predicted_points, p_start.
        """
        self.df = predictions_df.copy()
        self._validate_columns()
    
    def _validate_columns(self) -> None:
        """Ensure required columns exist."""
        required = ["player_id", "predicted_points", "p_start"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def get_recommendation(
        self,
        squad_ids: Optional[List[int]] = None,
    ) -> Optional[CaptainRecommendation]:
        """Get captain recommendation using argmax(predicted_points).
        
        Args:
            squad_ids: List of player IDs in user's squad.
                If None, considers all players.
        
        Returns:
            CaptainRecommendation for the highest predicted_points player.
        """
        # Filter to squad if provided
        if squad_ids:
            candidates = self.df[self.df["player_id"].isin(squad_ids)]
        else:
            candidates = self.df
        
        if candidates.empty:
            return None
        
        # Frozen rule: argmax(predicted_points)
        best = candidates.nlargest(1, "predicted_points").iloc[0]
        return self._row_to_recommendation(best)
    
    def get_recommendations(
        self,
        squad_ids: Optional[List[int]] = None,
    ) -> List[CaptainRecommendation]:
        """Get captain recommendation as a list (backward compatibility).
        
        Args:
            squad_ids: List of player IDs in user's squad.
        
        Returns:
            List containing single CaptainRecommendation.
        """
        pick = self.get_recommendation(squad_ids)
        return [pick] if pick else []
    
    def _row_to_recommendation(self, row: pd.Series) -> CaptainRecommendation:
        """Convert DataFrame row to CaptainRecommendation."""
        return CaptainRecommendation(
            player_id=int(row["player_id"]),
            player_name=str(row.get("player_name", "Unknown")),
            team_name=str(row.get("team_name", "")),
            ev_doubled=float(row["predicted_points"]) * 2,
            p_start=float(row["p_start"]),
        )
    
    def print_recommendation(self, pick: CaptainRecommendation) -> None:
        """Print formatted captain recommendation."""
        print("\n" + "=" * 70)
        print("CAPTAIN RECOMMENDATION")
        print("=" * 70)
        print(f"\n🎯 {pick.player_name} ({pick.team_name})")
        print(f"   EV×2: {pick.ev_doubled:.1f} | P(start): {pick.p_start:.0%}")
        print("\n" + "=" * 70)
