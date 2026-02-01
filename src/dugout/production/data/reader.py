"""Read-only access layer for the FPL SQLite database.

Provides DataReader class for querying the local SQLite database containing
FPL player history, teams, and fixtures. All methods are read-only.

Key Methods:
    query() - Execute raw SQL and return list of dicts
    get_player() - Get player by ID with optional history
    get_all_gw_data() - Get all gameweek history joined with player info
    get_team_name() - Get team name by ID

Usage:
    from dugout.production.data import DataReader
    
    reader = DataReader()
    df = reader.get_all_gw_data()
    player = reader.get_player(player_id=1, include_history=True)
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple, Any

import pandas as pd

from dugout.production.config import DEFAULT_DB_PATH
from dugout.production.data import queries as Q


def _dict_factory(cursor, row):
    mapping = {}
    for idx, col in enumerate(cursor.description):
        mapping[col[0]] = row[idx]
    return mapping


class DataReader:
    """Lightweight SQLite client for FPL data access."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self._history_cache: Dict[Tuple[int, int], List[Dict]] = {}

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = _dict_factory
        try:
            yield conn
        finally:
            conn.close()

    def query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute raw SQL and return list of dicts."""
        with self._conn() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall()

    # -------------------------------------------------------------------------
    # Player queries
    # -------------------------------------------------------------------------

    def get_player(self, player_id: int) -> Optional[Dict]:
        """Get player by ID."""
        rows = self.query(Q.PLAYER_BY_ID, (player_id,))
        return rows[0] if rows else None

    def get_players(
        self,
        status: Optional[str] = None,
        team_id: Optional[int] = None,
        position: Optional[int] = None,
    ) -> List[Dict]:
        """Get players with optional filters."""
        sql = Q.PLAYERS_BASE
        params = []

        if status:
            sql += " AND status = ?"
            params.append(status)
        if team_id:
            sql += " AND team = ?"
            params.append(team_id)
        if position:
            sql += " AND element_type = ?"
            params.append(position)

        return self.query(sql, tuple(params))

    def get_available_players(self) -> List[Dict]:
        """Get players available to play (not injured/suspended)."""
        return self.query(Q.PLAYERS_AVAILABLE)

    # -------------------------------------------------------------------------
    # Team queries
    # -------------------------------------------------------------------------

    def get_team(self, team_id: int) -> Optional[Dict]:
        """Get team by ID."""
        rows = self.query(Q.TEAM_BY_ID, (team_id,))
        return rows[0] if rows else None

    def get_teams(self) -> List[Dict]:
        """Get all teams."""
        return self.query(Q.TEAMS_ALL)

    # -------------------------------------------------------------------------
    # Fixture queries
    # -------------------------------------------------------------------------

    def get_fixtures(self, gw: Optional[int] = None) -> List[Dict]:
        """Get fixtures, optionally filtered by gameweek."""
        if gw:
            return self.query(Q.FIXTURES_BY_GW, (gw,))
        return self.query(Q.FIXTURES_ALL)

    def get_next_fixtures(self, team_id: int, n: int = 5) -> List[Dict]:
        """Get next N fixtures for a team."""
        return self.query(Q.FIXTURES_NEXT_FOR_TEAM, (team_id, team_id, n))

    # -------------------------------------------------------------------------
    # Gameweek data queries
    # -------------------------------------------------------------------------

    def get_player_history(self, player_id: int, last_n: int = 5) -> List[Dict]:
        """Get player's recent GW performances."""
        return self.query(Q.PLAYER_HISTORY, (player_id, last_n))

    def get_gw_data(self, gw: int) -> pd.DataFrame:
        """Get all player data for a specific gameweek."""
        return pd.DataFrame(self.query(Q.GW_DATA_SINGLE, (gw,)))

    def get_current_gw(self) -> Optional[int]:
        """Get the latest gameweek with data."""
        rows = self.query(Q.CURRENT_GW)
        return rows[0]["gw"] if rows and rows[0]["gw"] else None

    def get_all_gw_data(self) -> pd.DataFrame:
        """Get all gameweek data with player/team info.
        
        Returns DataFrame with player_id, player_name, gw, total_points, etc.
        """
        return pd.DataFrame(self.query(Q.GW_DATA_FULL))

    def _add_opponent_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add opponent_team and is_home columns from fixtures."""
        opponent_map = self.get_opponent_map()
        
        def lookup_opponent(row):
            key = (row["team_id"], row["round"])
            return opponent_map.get(key, (None, None))
        
        results = df.apply(lookup_opponent, axis=1, result_type="expand")
        df["opponent_team"] = results[0]
        df["is_home"] = results[1]
        return df

    def get_opponent_map(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Build lookup: (team_id, gw) -> (opponent_id, is_home).
        
        Returns:
            Dict mapping (team_id, round) to (opponent_team_id, is_home)
        """
        fixtures = self.get_fixtures()
        opponent_map = {}
        for f in fixtures:
            gw = f["event"]
            if gw is None:
                continue
            # Home team faces away team
            opponent_map[(f["team_h"], gw)] = (f["team_a"], 1)
            # Away team faces home team
            opponent_map[(f["team_a"], gw)] = (f["team_h"], 0)
        return opponent_map

    def get_fixture_display_map(self, gw: int) -> Dict[int, Dict[str, Any]]:
        """Get fixture display info for a gameweek (opponent_short, is_home).
        
        This is for DISPLAY ONLY - not used in predictions or rankings.
        
        Args:
            gw: Target gameweek
            
        Returns:
            Dict mapping team_id -> {"opponent_short": str, "is_home": bool}
        """
        fixtures = self.get_fixtures(gw=gw)
        teams = self.get_teams()
        team_short = {t["id"]: t["short_name"] for t in teams}
        
        fixture_map = {}
        for f in fixtures:
            h, a = f["team_h"], f["team_a"]
            fixture_map[h] = {"opponent_short": team_short.get(a, "?"), "is_home": True}
            fixture_map[a] = {"opponent_short": team_short.get(h, "?"), "is_home": False}
        return fixture_map

    def enrich_with_fixture_display(
        self,
        df: pd.DataFrame,
        gw: int,
        team_col: str = "team_id",
    ) -> pd.DataFrame:
        """Add opponent_short and is_home columns for display only.
        
        Args:
            df: DataFrame with team column
            gw: Target gameweek for fixtures
            team_col: Column name containing team_id (default: "team_id")
            
        Returns:
            DataFrame with opponent_short and is_home columns added
        """
        fixture_map = self.get_fixture_display_map(gw)
        df = df.copy()
        df["opponent_short"] = df[team_col].map(
            lambda x: fixture_map.get(x, {}).get("opponent_short", "?")
        )
        df["is_home"] = df[team_col].map(
            lambda x: fixture_map.get(x, {}).get("is_home", False)
        )
        return df

    def is_home_fixture(self, team_id: int, gw: int) -> int:
        """Check if team plays at home in given GW.
        
        Args:
            team_id: Team ID to check
            gw: Gameweek number
            
        Returns:
            1 if home, 0 if away
            
        Raises:
            KeyError: If no fixture found for team_id/gw (blank GW)
        """
        fixture = self.get_opponent_map().get((team_id, gw))
        if fixture is None:
            raise KeyError(f"No fixture found for team {team_id} in GW {gw}")
        return fixture[1]

    # -------------------------------------------------------------------------
    # Bulk history for feature engineering
    # -------------------------------------------------------------------------

    def get_players_history_bulk(
        self,
        player_ids: Iterable[int],
        last_n: int = 5,
    ) -> Dict[int, List[Dict]]:
        """Get recent history for multiple players efficiently."""
        unique_ids = tuple(set(player_ids))
        if not unique_ids:
            return {}

        placeholders = ",".join("?" for _ in unique_ids)
        sql = Q.PLAYERS_HISTORY_BULK.format(placeholders=placeholders)
        rows = self.query(sql, (*unique_ids, last_n))

        result: Dict[int, List[Dict]] = {pid: [] for pid in unique_ids}
        for row in rows:
            result[row["element_id"]].append(row)

        return result
