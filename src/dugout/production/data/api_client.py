"""FPL API client for HTTP requests.

Handles all communication with the official FPL API including:
- Rate limiting with adaptive delays
- Retry logic with exponential backoff
- Response caching for bootstrap data

Key Classes:
    FPLApiClient - HTTP client for FPL API

Usage:
    from dugout.production.data.api_client import FPLApiClient
    
    client = FPLApiClient()
    bootstrap = client.get_bootstrap()
    fixtures = client.get_fixtures()
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from dugout.production.config import FPL_BASE_URL, REQUEST_DELAY, MAX_RETRIES

logger = logging.getLogger(__name__)

# FPL API endpoints
ENDPOINTS = {
    "bootstrap": f"{FPL_BASE_URL}/bootstrap-static/",
    "fixtures": f"{FPL_BASE_URL}/fixtures/",
    "live": f"{FPL_BASE_URL}/event/{{gw}}/live/",
    "player": f"{FPL_BASE_URL}/element-summary/{{player_id}}/",
}

# Rate limiting constants (not in config - implementation details)
MAX_DELAY = 60  # Maximum backoff delay in seconds
RATE_LIMIT_STATUS = 429  # HTTP "Too Many Requests"


class FPLApiClient:
    """HTTP client for FPL API with rate limiting and caching."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/537.36"
        })
        self._bootstrap_cache: Optional[Dict] = None
        self._current_delay = REQUEST_DELAY
    
    def _get(self, url: str, retries: int = MAX_RETRIES) -> Optional[Dict]:
        """Make GET request with retry logic and adaptive rate limiting."""
        for attempt in range(retries):
            try:
                # Add jitter to prevent thundering herd; when many clients retry at the same time
                jitter = random.uniform(0, 0.3 * self._current_delay)
                time.sleep(self._current_delay + jitter)
                
                # Wait for the server 30 seconds timeout
                resp = self.session.get(url, timeout=30)
                
                # Handle rate limiting (429)
                if resp.status_code == RATE_LIMIT_STATUS:
                    retry_after = int(resp.headers.get("Retry-After", 30))

                    # exponential backoff; wait time till retry
                    self._current_delay = min(self._current_delay * 2, MAX_DELAY)
                    logger.warning(
                        f"Rate limited (429). Waiting {retry_after}s. "
                        f"Delay now: {self._current_delay}s"
                    )
                    time.sleep(retry_after)
                    continue
                
                resp.raise_for_status()
                
                # Success - gradually reduce delay back to base
                self._current_delay = max(REQUEST_DELAY, self._current_delay * 0.9)
                return resp.json()
                
            except requests.RequestException as e:
                wait_time = min(2 ** attempt + random.uniform(0, 1), MAX_DELAY)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{retries}): {e}. "
                    f"Retrying in {wait_time:.1f}s"
                )
                if attempt < retries - 1:
                    time.sleep(wait_time)
        
        logger.error(f"All {retries} attempts failed for {url}")
        return None
    
    def get_bootstrap(self, force: bool = False) -> Dict:
        """Get bootstrap-static data (cached).
        
        Args:
            force: If True, bypass cache and fetch fresh data.
            
        Returns:
            Bootstrap data dict.
            
        Raises:
            RuntimeError: If bootstrap data cannot be fetched.
        """
        if self._bootstrap_cache is None or force:
            logger.info("Fetching bootstrap-static data...")
            self._bootstrap_cache = self._get(ENDPOINTS["bootstrap"])
        
        if self._bootstrap_cache is None:
            raise RuntimeError("Failed to fetch bootstrap data from FPL API")
        
        return self._bootstrap_cache
    
    def get_current_gw(self) -> int:
        """Get the current/latest finished gameweek.
        
        Returns:
            Current gameweek number.
            
        Raises:
            RuntimeError: If bootstrap data cannot be fetched.
        """
        logger.info("Getting current gameweek...")
        bootstrap = self.get_bootstrap()
        events = bootstrap.get("events", [])
        
        # Priority 1: Find the current gameweek
        for event in events:
            if event.get("is_current"):
                return event["id"]
        
        # Priority 2: If no current, use next - 1
        for event in events:
            if event.get("is_next"):
                return event["id"] - 1
        
        # Fallback: find latest finished
        finished = [e for e in bootstrap.get("events", []) if e.get("finished")]
        if finished:
            return max(e["id"] for e in finished)
        
        raise RuntimeError("No gameweek data found in bootstrap")
    
    def get_gw_deadline(self, gw: int) -> Optional[datetime]:
        """Get deadline for a specific gameweek."""
        logger.info(f"Getting GW{gw} deadline...")
        bootstrap = self.get_bootstrap()
        for event in bootstrap.get("events", []):
            if event["id"] == gw:
                deadline_str = event.get("deadline_time")
                if deadline_str:
                    return datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
        return None
    
    def get_gw(self, gw: int) -> Optional[Dict]:
        """Get player stats for a gameweek.
        
        Works for any gameweek - finished, in-progress, or upcoming.
        """
        url = ENDPOINTS["live"].format(gw=gw)
        logger.info(f"Fetching GW{gw} data...")
        return self._get(url)
    
    def get_fixtures(self) -> Optional[List[Dict]]:
        """Get all fixtures."""
        logger.info("Fetching fixtures...")
        return self._get(ENDPOINTS["fixtures"])
    
    def get_player_history(self, player_id: int) -> Optional[Dict]:
        """Get player's detailed history."""
        logger.info(f"Fetching player {player_id} history...")
        url = ENDPOINTS["player"].format(player_id=player_id)
        return self._get(url)
    
    def is_gw_finished(self, gw: int) -> bool:
        """Check if a gameweek has finished (all matches complete, bonus confirmed).
        
        Uses FPL's official 'finished' flag rather than time-based heuristics.
        """
        logger.info(f"Checking if GW{gw} is finished...")
        bootstrap = self.get_bootstrap()
        for event in bootstrap.get("events", []):
            if event["id"] == gw:
                return event.get("finished", False)
        return False
