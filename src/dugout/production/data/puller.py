"""FPL data puller - orchestrates API and database operations.

Fetches latest data from the official FPL API and updates the local
SQLite database. Designed to run ~5 hours after each gameweek deadline
to capture all bonus points and final scores.

This module provides a facade over FPLApiClient and FPLDatabaseManager
for backwards compatibility and ease of use.

Key Classes:
    FPLPuller - Main orchestrator class

Usage:
    from dugout.production.data import FPLPuller
    
    puller = FPLPuller()
    puller.update_database(gw=20)

CLI:
    python scripts/pull_fpl_data.py --gw 18
    python scripts/pull_fpl_data.py --full --force
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from dugout.production.config import DEFAULT_DB_PATH
from dugout.production.data.api_client import FPLApiClient
from dugout.production.data.db_manager import FPLDatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FPLPuller:
    """Orchestrates FPL data fetching and storage.
    
    Combines FPLApiClient (HTTP) and FPLDatabaseManager (SQLite) to provide
    a simple interface for updating local FPL data.
    
    Example:
        puller = FPLPuller()
        puller.update_database(gw=20)
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize puller with API client and database manager.
        
        Args:
            db_path: Path to SQLite database. Uses default if not specified.
        """
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.api = FPLApiClient()
        self.db = FPLDatabaseManager(str(self.db_path))
    
    # === Delegate API methods for backwards compatibility ===
    
    def get_bootstrap(self, force: bool = False):
        """Get bootstrap-static data (cached)."""
        return self.api.get_bootstrap(force)
    
    def get_current_gw(self):
        """Get the current/latest finished gameweek."""
        return self.api.get_current_gw()
    
    def get_gw_deadline(self, gw: int):
        """Get deadline for a specific gameweek."""
        return self.api.get_gw_deadline(gw)
    
    def get_gw(self, gw: int):
        """Get player stats for a gameweek."""
        return self.api.get_gw(gw)
    
    def get_fixtures(self):
        """Get all fixtures."""
        return self.api.get_fixtures()
    
    def get_player_history(self, player_id: int):
        """Get player's detailed history."""
        return self.api.get_player_history(player_id)
    
    def is_gw_finished(self, gw: int) -> bool:
        """Check if a gameweek has finished."""
        return self.api.is_gw_finished(gw)
    
    # === Main orchestration method ===
    
    def update_database(self, gw: Optional[int] = None, full: bool = False) -> bool:
        """Update the database with latest data.
        
        Args:
            gw: Specific gameweek to update (None = current). 
                Mutually exclusive with full=True.
            full: If True, pull ALL finished gameweeks (initial setup).
                Mutually exclusive with gw.
            
        Returns:
            True if successful, False otherwise.
            
        Raises:
            RuntimeError: If bootstrap data cannot be fetched.
            ValueError: If both gw and full are specified.
        """
        if gw is not None and full:
            raise ValueError("Cannot specify both gw and full=True. Use one or the other.")
        
        bootstrap = self.api.get_bootstrap()  # Raises RuntimeError on failure
        fixtures = self.api.get_fixtures()
        conn = self.db.get_connection()
        
        try:
            # Update players/teams (always current from API)
            self.db.update_players(conn, bootstrap)
            self.db.update_teams(conn, bootstrap)
            
            # Update ALL fixtures (schedule + results)
            self.db.update_fixtures(conn, fixtures)
            
            # Determine which GWs to pull
            if full:
                finished_gws = [
                    e["id"] for e in bootstrap.get("events", []) 
                    if e.get("finished")
                ]
                logger.info(f"Full pull: {len(finished_gws)} finished gameweeks")
            else:
                target_gw = gw or self.api.get_current_gw()
                if not target_gw:
                    logger.error("Could not determine current gameweek")
                    return False
                finished_gws = [target_gw]
            
            # Pull each gameweek
            for gw_num in finished_gws:
                logger.info(f"Pulling GW{gw_num}...")
                live_data = self.api.get_gw(gw_num)
                if not live_data:
                    logger.warning(f"Failed to fetch GW{gw_num} live data, skipping")
                    continue
                
                self.db.update_gameweek_data(conn, bootstrap, live_data, gw_num)
            
            conn.commit()
            logger.info(f"✅ Database updated successfully ({len(finished_gws)} GWs)")
            return True
            
        except Exception as e:
            logger.error(f"Database update failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
