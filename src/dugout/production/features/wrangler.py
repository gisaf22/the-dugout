"""Data wrangling utilities for FPL data preparation.

Provides the Wrangler class for cleaning and formatting raw FPL data.
Single-purpose: handles deduplication and column normalization.
Scaling/normalization delegated to separate FeatureScaler class.

Key Methods:
    get_prediction_ready_data() - Master orchestrator for data cleaning pipeline
    normalize_position() - Normalize position strings to canonical format

Usage:
    from dugout.production.features import Wrangler
    
    wrangler = Wrangler()
    df = wrangler.get_prediction_ready_data()  # Training data with full history
"""

from __future__ import annotations

from typing import List, Optional, Set

import pandas as pd

from dugout.production.data.reader import DataReader
from dugout.production.config import ELEMENT_TYPE_TO_POS


class Wrangler:
    """Data preparation and transformation utilities."""
    
    def __init__(self, reader: Optional[DataReader] = None):
        self.reader = reader or DataReader()

    def get_prediction_ready_data(self) -> pd.DataFrame:
        """Master orchestrator: load, clean, and format player data for ML.
        
        Coordinates cleaning pipeline:
        1. Load all gameweek history
        2. Remove duplicate rows
        3. Format all columns to ML-ready types
        
        Returns:
            Clean, deduplicated, formatted DataFrame ready for feature engineering.
            Includes all historical player data (no status filtering).
        """
        df = self._load_all_gw_data()
        df = self._remove_duplicates(df)
        df = self._format_columns(df)
        return df
    
    def _load_all_gw_data(self) -> pd.DataFrame:
        """Load all gameweek history from data reader."""
        return self.reader.get_all_gw_data()
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows (same player, same gameweek).
        
        Single purpose: deduplication only.
        
        Returns:
            DataFrame with duplicates removed (keeps first occurrence).
        """
        return df.drop_duplicates(subset=["element_id", "round"], keep="first")
    
    def _format_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format all columns to ML-ready types.
        
        Applies column-specific conversions:
        - element_type → position (1,2,3,4 → GKP,DEF,MID,FWD), then drops element_type
        - Drops season_id (constant for current season, no predictive value)
        
        Returns:
            DataFrame with formatted columns.
        """
        df = df.copy()
        
        # Convert position codes and drop redundant element_type
        if "element_type" in df.columns:
            df["position"] = df["element_type"].map(ELEMENT_TYPE_TO_POS)
            df = df.drop(columns=["element_type"])
        
        # Drop constant season_id (always 10 for current season)
        if "season_id" in df.columns:
            df = df.drop(columns=["season_id"])
        
        return df

    @staticmethod
    def normalize_position(pos: str) -> str:
        """Normalize position string to FPL standard format.
        
        Converts various position name formats to canonical FPL position codes.
        Use when loading external data or user input to ensure consistent position encoding.
        
        Args:
            pos: Position string in any format (e.g., 'GK', 'goalkeeper', 'DEF', 'Midfielder').
        
        Returns:
            Normalized position code: 'GKP', 'DEF', 'MID', or 'FWD'.
            Returns input as-is if format not recognized.
        """
        pos = pos.upper().strip()
        if pos in ("GK", "GKP", "GOALKEEPER"):
            return "GKP"
        if pos in ("DEF", "DEFENDER"):
            return "DEF"
        if pos in ("MID", "MIDFIELDER"):
            return "MID"
        if pos in ("FWD", "FORWARD", "ATT", "ATTACKER"):
            return "FWD"
        return pos
