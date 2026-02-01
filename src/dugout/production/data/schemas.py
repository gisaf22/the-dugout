"""Pydantic schemas for FPL data validation.

Defines Pydantic models for FPL API responses and database rows.
Used for type safety and data validation throughout the codebase.

Models:
    PlayerSchema - FPL player with position/cost properties
    TeamSchema - FPL team info (name, short_name, strength)
    FixtureSchema - Match fixture with home/away teams
    HistorySchema - Player gameweek history row

Usage:
    from dugout.production.data.schemas import PlayerSchema
    
    player = PlayerSchema.model_validate(api_response)
    print(player.position, player.cost_millions)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, get_args, get_origin

from pydantic import BaseModel, Field

from dugout.production.config import ELEMENT_TYPE_TO_POS


# Type mapping from Python types to SQLite types
PYTHON_TO_SQLITE: Dict[Type, str] = {
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    bool: "INTEGER",
}


def pydantic_to_sqlite_column(field_name: str, field_info: Any) -> str:
    """Convert a Pydantic field to SQLite column definition.
    
    Args:
        field_name: Name of the field
        field_info: Pydantic FieldInfo object
        
    Returns:
        SQLite column definition string
    """
    annotation = field_info.annotation
    
    # Handle Optional[X] -> extract X
    origin = get_origin(annotation)
    if origin is type(None) or annotation is type(None):
        return f"{field_name} TEXT"
    
    # Optional is Union[X, None]
    if hasattr(annotation, "__origin__"):
        args = get_args(annotation)
        if type(None) in args:
            # Get the non-None type
            annotation = next(a for a in args if a is not type(None))
    
    sqlite_type = PYTHON_TO_SQLITE.get(annotation, "TEXT")
    
    # Primary key handling
    if field_name == "id":
        return f"{field_name} {sqlite_type} PRIMARY KEY"
    
    return f"{field_name} {sqlite_type}"


def schema_to_create_table(
    table_name: str,
    schema: Type[BaseModel],
    extra_columns: Optional[List[str]] = None,
    unique_constraint: Optional[str] = None,
) -> str:
    """Generate CREATE TABLE SQL from Pydantic schema.
    
    Args:
        table_name: Name of the SQL table
        schema: Pydantic model class
        extra_columns: Additional column definitions not in schema
        unique_constraint: Optional UNIQUE constraint clause
        
    Returns:
        CREATE TABLE IF NOT EXISTS SQL statement
    """
    columns = []
    
    for field_name, field_info in schema.model_fields.items():
        columns.append(pydantic_to_sqlite_column(field_name, field_info))
    
    # Add extra columns (e.g., season_id, modified)
    if extra_columns:
        columns.extend(extra_columns)
    
    if unique_constraint:
        columns.append(unique_constraint)
    
    columns_sql = ",\n                ".join(columns)
    return f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {columns_sql}
            )
        """


class PlayerSchema(BaseModel):
    """FPL Player data."""
    
    id: int
    first_name: Optional[str] = None
    second_name: Optional[str] = None
    web_name: Optional[str] = None
    team: Optional[int] = None
    element_type: Optional[int] = None
    now_cost: Optional[int] = None
    status: Optional[str] = None
    chance_of_playing_next_round: Optional[int] = None
    total_points: Optional[int] = None
    form: Optional[float] = None
    points_per_game: Optional[float] = None
    selected_by_percent: Optional[float] = None

    @property
    def position(self) -> str:
        """Get position string (GKP, DEF, MID, FWD)."""
        return ELEMENT_TYPE_TO_POS.get(self.element_type, "UNK")

    @property
    def cost_millions(self) -> float:
        """Get cost in millions (e.g., 10.5)."""
        return (self.now_cost or 0) / 10.0

    @property
    def display_name(self) -> str:
        """Get best available display name."""
        return self.web_name or f"{self.first_name} {self.second_name}"


class TeamSchema(BaseModel):
    """FPL Team data."""
    
    id: int
    name: Optional[str] = None
    short_name: Optional[str] = None
    strength: Optional[int] = None
    strength_overall_home: Optional[int] = None
    strength_overall_away: Optional[int] = None
    strength_attack_home: Optional[int] = None
    strength_attack_away: Optional[int] = None
    strength_defence_home: Optional[int] = None
    strength_defence_away: Optional[int] = None


class FixtureSchema(BaseModel):
    """FPL Fixture data."""
    
    id: int
    event: Optional[int] = Field(None, alias="gameweek")
    team_h: Optional[int] = None
    team_a: Optional[int] = None
    team_h_score: Optional[int] = None
    team_a_score: Optional[int] = None
    kickoff_time: Optional[str] = None
    finished: Optional[bool] = None
    
    class Config:
        populate_by_name = True


class GameweekSchema(BaseModel):
    """Player performance for a single gameweek."""
    
    element_id: int  # Player ID (FK to players.id)
    round: int
    minutes: int = 0
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    bonus: int = 0
    bps: int = 0
    total_points: int = 0
    influence: float = 0.0
    creativity: float = 0.0
    threat: float = 0.0
    ict_index: float = 0.0
    expected_goals: float = 0.0
    expected_assists: float = 0.0
    starts: int = 0


class PredictionSchema(BaseModel):
    """Model prediction output."""
    
    player_id: int
    player_name: str
    team_name: str
    position: str
    predicted_points: float
    uncertainty: float = 0.0
    minutes_fraction: float = 0.5
    p_haul: float = 0.0
    cost: float = 5.0


# Backwards compatibility aliases
Player = PlayerSchema
Team = TeamSchema
Fixture = FixtureSchema
GameweekEntry = GameweekSchema
