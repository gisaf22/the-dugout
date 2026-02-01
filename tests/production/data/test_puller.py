"""Tests for FPL data puller module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from dugout.production.data.puller import FPLPuller
from dugout.production.data.api_client import FPLApiClient
from dugout.production.data.db_manager import FPLDatabaseManager


class TestFPLPuller:
    """Tests for FPLPuller orchestrator class."""
    
    def test_init_creates_api_and_db(self, tmp_path):
        """Puller initializes with API client and DB manager."""
        db_path = tmp_path / "test.sqlite"
        puller = FPLPuller(db_path=str(db_path))
        
        assert isinstance(puller.api, FPLApiClient)
        assert isinstance(puller.db, FPLDatabaseManager)
        assert puller.db_path == db_path
    
    def test_delegates_to_api_client(self, tmp_path):
        """API methods delegate to FPLApiClient."""
        db_path = tmp_path / "test.sqlite"
        puller = FPLPuller(db_path=str(db_path))
        
        # Mock the API client
        puller.api = Mock(spec=FPLApiClient)
        puller.api.get_bootstrap.return_value = {"events": []}
        puller.api.get_current_gw.return_value = 20
        puller.api.get_fixtures.return_value = []
        puller.api.is_gw_finished.return_value = True
        
        # Call delegated methods
        assert puller.get_bootstrap() == {"events": []}
        assert puller.get_current_gw() == 20
        assert puller.get_fixtures() == []
        assert puller.is_gw_finished(20) is True
        
        # Verify delegation
        puller.api.get_bootstrap.assert_called_once()
        puller.api.get_current_gw.assert_called_once()
        puller.api.get_fixtures.assert_called_once()
        puller.api.is_gw_finished.assert_called_once_with(20)
    
    def test_update_database_raises_on_bootstrap_failure(self, tmp_path):
        """Raises RuntimeError if bootstrap fetch fails."""
        db_path = tmp_path / "test.sqlite"
        puller = FPLPuller(db_path=str(db_path))
        puller.api = Mock(spec=FPLApiClient)
        puller.api.get_bootstrap.side_effect = RuntimeError("Failed to fetch bootstrap")
        
        with pytest.raises(RuntimeError, match="Failed to fetch bootstrap"):
            puller.update_database(gw=10)
    
    def test_update_database_success_flow(self, tmp_path):
        """Successful update calls all expected methods."""
        db_path = tmp_path / "test.sqlite"
        puller = FPLPuller(db_path=str(db_path))
        
        # Mock API
        puller.api = Mock(spec=FPLApiClient)
        puller.api.get_bootstrap.return_value = {
            "events": [{"id": 10, "finished": True}],
            "elements": [],
            "teams": []
        }
        puller.api.get_fixtures.return_value = []
        puller.api.get_current_gw.return_value = 10
        puller.api.get_gw.return_value = {"elements": []}
        
        # Mock DB
        mock_conn = MagicMock()
        puller.db = Mock(spec=FPLDatabaseManager)
        puller.db.get_connection.return_value = mock_conn
        
        result = puller.update_database(gw=10)
        
        assert result is True
        puller.db.update_players.assert_called_once()
        puller.db.update_teams.assert_called_once()
        puller.db.update_fixtures.assert_called_once()
        puller.db.update_gameweek_data.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()


class TestFPLApiClient:
    """Tests for FPLApiClient HTTP client."""
    
    def test_bootstrap_cache(self):
        """Bootstrap data is cached after first fetch."""
        client = FPLApiClient()
        client._bootstrap_cache = {"cached": True}
        
        result = client.get_bootstrap()
        
        assert result == {"cached": True}
    
    def test_get_current_gw_from_is_current(self):
        """Extracts current GW from events with is_current flag."""
        client = FPLApiClient()
        client._bootstrap_cache = {
            "events": [
                {"id": 18, "is_current": False, "finished": True},
                {"id": 19, "is_current": True, "finished": False},
                {"id": 20, "is_next": True, "finished": False},
            ]
        }
        
        assert client.get_current_gw() == 19
    
    def test_get_current_gw_from_is_next(self):
        """Falls back to is_next - 1 if no is_current."""
        client = FPLApiClient()
        client._bootstrap_cache = {
            "events": [
                {"id": 18, "finished": True},
                {"id": 19, "finished": True},
                {"id": 20, "is_next": True, "finished": False},
            ]
        }
        
        assert client.get_current_gw() == 19


class TestFPLDatabaseManager:
    """Tests for FPLDatabaseManager database operations."""
    
    def test_init_creates_database(self, tmp_path):
        """Database file is created on init."""
        db_path = tmp_path / "subdir" / "test.sqlite"
        
        db = FPLDatabaseManager(str(db_path))
        
        assert db_path.exists()
    
    def test_tables_created(self, tmp_path):
        """All required tables are created."""
        import sqlite3
        
        db_path = tmp_path / "test.sqlite"
        db = FPLDatabaseManager(str(db_path))
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        assert "players" in tables
        assert "teams" in tables
        assert "fixtures" in tables
        assert "gameweeks" in tables
