"""Database manager for FPL data storage.

Handles all SQLite database operations including:
- Schema initialization and migrations
- CRUD operations for players, teams, fixtures, gameweeks

Key Classes:
    FPLDatabaseManager - Database operations for FPL data

Usage:
    from dugout.production.data.db_manager import FPLDatabaseManager
    
    db = FPLDatabaseManager()
    db.update_players(bootstrap_data)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from dugout.production.config import CURRENT_SEASON_ID, DEFAULT_DB_PATH
from dugout.production.data.schemas import (
    PlayerSchema,
    TeamSchema,
    FixtureSchema,
    GameweekSchema,
    schema_to_create_table,
)
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class FPLDatabaseManager:
    """Manages SQLite database for FPL data storage."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self._init_database()
    
    def _init_database(self) -> None:
        """Create database schema if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        self._create_players_table(cursor)
        self._create_teams_table(cursor)
        self._create_fixtures_table(cursor)
        self._create_gameweeks_table(cursor)
        self._create_indexes(cursor)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def _create_players_table(self, cursor: sqlite3.Cursor) -> None:
        """Create players table from PlayerSchema."""
        sql = schema_to_create_table(
            "players",
            PlayerSchema,
            extra_columns=[f"season_id INTEGER DEFAULT {CURRENT_SEASON_ID}", "modified TEXT"],
        )
        cursor.execute(sql)
    
    def _create_teams_table(self, cursor: sqlite3.Cursor) -> None:
        """Create teams table from TeamSchema."""
        sql = schema_to_create_table(
            "teams",
            TeamSchema,
            extra_columns=[f"season_id INTEGER DEFAULT {CURRENT_SEASON_ID}"],
        )
        cursor.execute(sql)
    
    def _create_fixtures_table(self, cursor: sqlite3.Cursor) -> None:
        """Create fixtures table from FixtureSchema."""
        sql = schema_to_create_table(
            "fixtures",
            FixtureSchema,
            extra_columns=[f"season_id INTEGER DEFAULT {CURRENT_SEASON_ID}"],
        )
        cursor.execute(sql)
    
    def _create_gameweeks_table(self, cursor: sqlite3.Cursor) -> None:
        """Create gameweeks table from GameweekSchema + extra columns."""
        sql = schema_to_create_table(
            "gameweeks",
            GameweekSchema,
            extra_columns=[
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                f"season_id INTEGER DEFAULT {CURRENT_SEASON_ID}",
                "GW INTEGER",
                "player_id INTEGER",
                "own_goals INTEGER",
                "penalties_saved INTEGER",
                "penalties_missed INTEGER",
                "yellow_cards INTEGER",
                "red_cards INTEGER",
                "saves INTEGER",
                "expected_goal_involvements REAL",
                "expected_goals_conceded REAL",
                "team_id INTEGER",
                "element_type INTEGER",
                "modified TEXT",
                "UNIQUE(element_id, round, season_id)",
            ],
        )
        cursor.execute(sql)
    
    def _create_indexes(self, cursor: sqlite3.Cursor) -> None:
        """Create database indexes for common query patterns."""
        # Composite indexes for frequent queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_gw_element_round 
            ON gameweeks(element_id, round)
        """)  # Player's GW history
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_gw_element_season 
            ON gameweeks(element_id, season_id)
        """)  # Player's season history
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_gw_round_season 
            ON gameweeks(round, season_id)
        """)  # All players in a GW
        
        # Covering index for predictions (avoids table lookup)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_gw_prediction 
            ON gameweeks(element_id, round, total_points, minutes)
        """)
        
        # Players table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_players_team 
            ON players(team)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_players_position 
            ON players(element_type)
        """)
        
        # Fixtures table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fixtures_event 
            ON fixtures(event)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fixtures_teams 
            ON fixtures(team_h, team_a)
        """)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    def _bulk_upsert(
        self,
        cursor: sqlite3.Cursor,
        table: str,
        columns: List[str],
        rows: List[tuple],
    ) -> None:
        """Bulk insert/replace rows efficiently."""
        if not rows:
            return
        placeholders = ", ".join("?" * len(columns))
        cols = ", ".join(columns)
        cursor.executemany(
            f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})",
            rows,
        )
    
    def update_players(self, conn: sqlite3.Connection, bootstrap: Dict) -> None:
        """Update players table from bootstrap data with schema validation."""
        raw_players = bootstrap.get("elements", [])
        logger.info(f"Validating and updating {len(raw_players)} players...")
        
        columns = [
            "id", "first_name", "second_name", "web_name", "team", "element_type",
            "now_cost", "status", "chance_of_playing_next_round",
            "total_points", "form", "points_per_game", "selected_by_percent",
            "season_id", "modified",
        ]
        rows = []
        errors = []
        
        for p in raw_players:
            try:
                player = PlayerSchema.model_validate(p)
                rows.append((
                    player.id, player.first_name, player.second_name,
                    player.web_name, player.team, player.element_type,
                    player.now_cost, player.status, player.chance_of_playing_next_round,
                    player.total_points, player.form, player.points_per_game,
                    player.selected_by_percent, CURRENT_SEASON_ID, "now",
                ))
            except ValidationError as e:
                errors.append((p.get("id", "unknown"), str(e)))
        
        if errors:
            logger.warning(f"Skipped {len(errors)} players with invalid schema: {errors[:3]}...")
        
        self._bulk_upsert(conn.cursor(), "players", columns, rows)
    
    def update_teams(self, conn: sqlite3.Connection, bootstrap: Dict) -> None:
        """Update teams table from bootstrap data with schema validation."""
        raw_teams = bootstrap.get("teams", [])
        logger.info(f"Validating and updating {len(raw_teams)} teams...")
        
        columns = [
            "id", "name", "short_name", "strength", "strength_overall_home",
            "strength_overall_away", "strength_attack_home", "strength_attack_away",
            "strength_defence_home", "strength_defence_away", "season_id",
        ]
        rows = []
        errors = []
        
        for t in raw_teams:
            try:
                team = TeamSchema.model_validate(t)
                rows.append((
                    team.id, team.name, team.short_name,
                    team.strength, team.strength_overall_home,
                    team.strength_overall_away, team.strength_attack_home,
                    team.strength_attack_away, team.strength_defence_home,
                    team.strength_defence_away, CURRENT_SEASON_ID,
                ))
            except ValidationError as e:
                errors.append((t.get("id", "unknown"), str(e)))
        
        if errors:
            logger.warning(f"Skipped {len(errors)} teams with invalid schema: {errors[:3]}...")
        
        self._bulk_upsert(conn.cursor(), "teams", columns, rows)
    
    def update_fixtures(self, conn: sqlite3.Connection, fixtures: List[Dict], gw: Optional[int] = None) -> None:
        """Update fixtures table with schema validation.
        
        Args:
            conn: Database connection
            fixtures: List of fixture dicts from API
            gw: If provided, only update fixtures for this GW. If None, update all fixtures.
        """
        if not fixtures:
            return
        
        if gw is not None:
            target_fixtures = [f for f in fixtures if f.get("event") == gw]
            logger.info(f"Validating and updating {len(target_fixtures)} fixtures for GW{gw}...")
        else:
            target_fixtures = fixtures
            logger.info(f"Validating and updating all {len(target_fixtures)} fixtures...")
        
        columns = [
            "id", "event", "team_h", "team_a", "team_h_score", "team_a_score",
            "finished", "kickoff_time", "season_id",
        ]
        rows = []
        errors = []
        
        for f in target_fixtures:
            try:
                fixture = FixtureSchema.model_validate(f)
                rows.append((
                    fixture.id, fixture.event, fixture.team_h, fixture.team_a,
                    fixture.team_h_score, fixture.team_a_score,
                    1 if fixture.finished else 0, fixture.kickoff_time, CURRENT_SEASON_ID,
                ))
            except ValidationError as e:
                errors.append((f.get("id", "unknown"), str(e)))
        
        if errors:
            logger.warning(f"Skipped {len(errors)} fixtures with invalid schema: {errors[:3]}...")
        
        self._bulk_upsert(conn.cursor(), "fixtures", columns, rows)
    
    def update_gameweek_data(
        self, 
        conn: sqlite3.Connection, 
        bootstrap: Dict, 
        live_data: Dict,
        gw: int
    ) -> None:
        """Update gameweeks table with player performance data.
        
        Note: Live data has nested 'stats' dict which we flatten.
        Validation happens on the flattened structure.
        """
        elements = live_data.get("elements", [])
        logger.info(f"Validating and updating {len(elements)} player performances for GW{gw}...")
        
        # Build player lookup for team_id and element_type
        player_lookup = {p["id"]: p for p in bootstrap.get("elements", [])}
        
        columns = [
            "element_id", "round", "season_id", "GW", "player_id",
            "minutes", "goals_scored", "assists", "clean_sheets",
            "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
            "yellow_cards", "red_cards", "saves", "bonus", "bps",
            "influence", "creativity", "threat", "ict_index",
            "total_points", "starts", "expected_goals", "expected_assists",
            "expected_goal_involvements", "expected_goals_conceded",
            "team_id", "element_type", "modified",
        ]
        
        rows = []
        errors = []
        
        for elem in elements:
            player_id = elem.get("id")
            if player_id is None:
                errors.append(("unknown", "Missing player id"))
                continue
                
            stats = elem.get("stats", {})
            player = player_lookup.get(player_id, {})
            
            # Flatten stats and validate against GameweekSchema
            flat_data = {
                "element_id": player_id,
                "round": gw,
                **stats,  # minutes, goals_scored, etc.
            }
            
            try:
                gw_entry = GameweekSchema.model_validate(flat_data)
                rows.append((
                    gw_entry.element_id, gw_entry.round, CURRENT_SEASON_ID, gw, player_id,
                    gw_entry.minutes,
                    gw_entry.goals_scored,
                    gw_entry.assists,
                    gw_entry.clean_sheets,
                    gw_entry.goals_conceded,
                    stats.get("own_goals", 0),
                    stats.get("penalties_saved", 0),
                    stats.get("penalties_missed", 0),
                    stats.get("yellow_cards", 0),
                    stats.get("red_cards", 0),
                    stats.get("saves", 0),
                    gw_entry.bonus,
                    gw_entry.bps,
                    gw_entry.influence,
                    gw_entry.creativity,
                    gw_entry.threat,
                    gw_entry.ict_index,
                    gw_entry.total_points,
                    gw_entry.starts,
                    gw_entry.expected_goals,
                    gw_entry.expected_assists,
                    stats.get("expected_goal_involvements", 0),
                    stats.get("expected_goals_conceded", 0),
                    player.get("team", 0),
                    player.get("element_type", 0),
                    "now",
                ))
            except ValidationError as e:
                errors.append((player_id, str(e)))
        
        if errors:
            logger.warning(f"Skipped {len(errors)} gameweek entries with invalid schema: {errors[:3]}...")
        
        self._bulk_upsert(conn.cursor(), "gameweeks", columns, rows)
