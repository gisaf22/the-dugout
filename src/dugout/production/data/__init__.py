"""Data module - data access, API, and schemas.

Public API:
    DataReader - SQLite database access
    FPLPuller - FPL API client (facade)
    FPLApiClient - HTTP client for FPL API
    FPLDatabaseManager - SQLite database operations
    PlayerSchema, FixtureSchema, etc. - Pydantic models
"""

from dugout.production.data.reader import DataReader
from dugout.production.data.puller import FPLPuller
from dugout.production.data.api_client import FPLApiClient
from dugout.production.data.db_manager import FPLDatabaseManager
from dugout.production.data.schemas import (
    PlayerSchema,
    FixtureSchema,
    GameweekSchema,
    TeamSchema,
    # Backwards compatibility aliases
    Player,
    Team,
    Fixture,
    GameweekEntry,
)

__all__ = [
    "DataReader",
    "FPLPuller",
    "FPLApiClient",
    "FPLDatabaseManager",
    "PlayerSchema",
    "FixtureSchema",
    "GameweekSchema",
    "TeamSchema",
    # Backwards compatibility aliases
    "Player",
    "Team",
    "Fixture", 
    "GameweekEntry",
]
