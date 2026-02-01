# The Dugout Data Integration Strategy
## SQLite Database Integration with fpl_data_collector

**Date**: November 19, 2025  
**Status**: Data Architecture Planning  

## Integration Overview

**Agent Goal Context**: The Data Agent's goal is to provide clean, fresh, feature-ready FPL data with low latency and high trust. Leveraging the existing `fpl_data_collector` SQLite database lets The Dugout concentrate engineering effort on reasoning & optimization layers while inheriting mature ingestion reliability.

The Dugout integrates with this database as a read-optimized, concurrency-aware source. We emphasize proactive freshness detection, anomaly surfacing, and strict Pydantic validation so downstream agents rely on stable semantics and confidence scoring.

## Expected Database Schema

Based on typical FPL data collection patterns, the SQLite database likely contains:

### Core Tables (Expected)

```sql
-- Player information and current stats
players (
    id INTEGER PRIMARY KEY,           -- FPL player ID
    first_name TEXT,
    second_name TEXT,
    web_name TEXT,                   -- Display name
    team_code INTEGER,               -- Team identifier
    element_type INTEGER,            -- Position (1=GK, 2=DEF, 3=MID, 4=FWD)
    now_cost INTEGER,                -- Current price in 0.1m units
    total_points INTEGER,            -- Season points total
    points_per_game REAL,           -- Average points per game
    selected_by_percent REAL,       -- Ownership percentage
    form REAL,                      -- Recent form score
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Gameweek-specific player performance
gameweek_data (
    id INTEGER PRIMARY KEY,
    player_id INTEGER,              -- Foreign key to players
    gameweek INTEGER,               -- Gameweek number
    total_points INTEGER,           -- Points scored this GW
    minutes INTEGER,                -- Minutes played
    goals_scored INTEGER,
    assists INTEGER,
    clean_sheets INTEGER,
    goals_conceded INTEGER,
    own_goals INTEGER,
    penalties_saved INTEGER,
    penalties_missed INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    saves INTEGER,
    bonus INTEGER,                  -- Bonus points
    bps INTEGER,                   -- Bonus point system score
    was_home BOOLEAN,              -- Home/away flag
    opponent_team INTEGER,         -- Opponent team code
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- Team information
teams (
    id INTEGER PRIMARY KEY,
    name TEXT,                     -- Team full name
    short_name TEXT,              -- 3-letter code
    strength_overall_home INTEGER, -- Team strength ratings
    strength_overall_away INTEGER,
    strength_attack_home INTEGER,
    strength_attack_away INTEGER,
    strength_defence_home INTEGER,
    strength_defence_away INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fixture information
fixtures (
    id INTEGER PRIMARY KEY,
    gameweek INTEGER,
    team_h INTEGER,               -- Home team ID
    team_a INTEGER,               -- Away team ID
    kickoff_time TIMESTAMP,
    team_h_difficulty INTEGER,   -- Difficulty rating for home team
    team_a_difficulty INTEGER,   -- Difficulty rating for away team
    team_h_score INTEGER,        -- Final score (null if not played)
    team_a_score INTEGER,
    finished BOOLEAN DEFAULT FALSE,
    minutes INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_h) REFERENCES teams(id),
    FOREIGN KEY (team_a) REFERENCES teams(id)
);

-- Price changes and transfer data
price_changes (
    id INTEGER PRIMARY KEY,
    player_id INTEGER,
    old_price INTEGER,
    new_price INTEGER,
    change_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- Gameweek metadata
gameweeks (
    id INTEGER PRIMARY KEY,
    name TEXT,                    -- "Gameweek 1", etc.
    deadline_time TIMESTAMP,     -- Transfer deadline
    finished BOOLEAN DEFAULT FALSE,
    is_current BOOLEAN DEFAULT FALSE,
    is_next BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Data Access Layer Design

### Data Reader (`src/fade/tools/data_reader.py`)

```python
"""
The Dugout Data Reader - SQLite Integration Layer

Provides lightweight, read-only access to the fpl_data_collector SQLite database
through small, composable query helpers. Designed for agent/tool consumption.
"""

class DataReader:
    """
    Lightweight helper for read-only SQLite operations.

    Responsibilities:
    - Open connections with WAL-friendly settings
    - Map rows to Pydantic models
    - Provide narrow, purpose-built query methods
    - Surface anomalies and missing data to orchestrators
    """

    def list_players_basic(self, limit: int = 50, offset: int = 0) -> List[Player]:
        """Retrieve basic player attributes (names, team, cost, status)."""

    def get_player_by_id(self, player_id: int) -> Optional[Player]:
        """Lookup a single player record by FPL ID."""

    def get_player_recent_history(self, player_id: int, last_n: int = 5) -> List[GameweekEntry]:
        """Return last N appearances for a player sorted by kickoff time."""

    def get_fixtures_by_gameweek(self, gw: int) -> List[Fixture]:
        """Fetch fixtures for a given gameweek ordered chronologically."""
```

### Data Models (`src/fade/models/fpl_schemas.py`)

```python
"""
The Dugout FPL Data Models - Pydantic Schemas

Defines standardized data structures for all FPL data used by The Dugout agents.
Ensures type safety and data validation across the system.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from enum import Enum

class PlayerPosition(Enum):
    GOALKEEPER = 1
    DEFENDER = 2  
    MIDFIELDER = 3
    FORWARD = 4

class PlayerData(BaseModel):
    """Current player information and season stats"""
    id: int = Field(..., description="FPL player ID")
    first_name: str
    second_name: str  
    web_name: str = Field(..., description="Display name")
    team_code: int
    position: PlayerPosition
    now_cost: float = Field(..., description="Current price in millions")
    total_points: int
    points_per_game: float
    selected_by_percent: float = Field(..., description="Ownership percentage")
    form: float = Field(..., description="Recent form score")
    updated_at: datetime

class GameweekPerformance(BaseModel):
    """Player performance in specific gameweek"""
    player_id: int
    gameweek: int
    total_points: int
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    bonus: int
    was_home: bool
    opponent_team: int
```

## Data Agent Implementation Strategy

### Core Responsibilities

1. **Database Connection Management**
   - Maintain connection pool for concurrent agent access
   - Handle connection failures and reconnection logic
   - Monitor database locks and optimize query patterns

2. **Data Standardization**
   - Convert raw database records to Pydantic models
   - Handle missing data and null value management
   - Ensure data consistency across agent requests

3. **Caching Strategy**
   - Cache frequently accessed data (current players, fixtures)
   - Implement cache invalidation on data updates
   - Balance memory usage with query performance

4. **Query Optimization**
   - Pre-built query patterns for common agent requests
   - Database indexing recommendations
   - Efficient join patterns for complex queries

### Agent Interface Design

```python
class DataAgent(BaseAgent):
    """
    The Dugout Data Agent - SQLite Database Interface
    
    Provides standardized data access for all The Dugout agents while optimizing
    database performance and ensuring data consistency.
    """
    
    async def get_player_data(self, player_ids: Optional[List[int]] = None) -> List[PlayerData]:
        """Get current player information"""
        
    async def get_performance_history(self, player_id: int, gameweeks: int = 10) -> List[GameweekPerformance]:
        """Get historical performance data for ML training"""
        
    async def get_fixture_difficulty(self, team_id: int, gameweeks: int = 5) -> List[FixtureDifficulty]:
        """Get upcoming fixture difficulty for team"""
        
    async def monitor_price_changes(self) -> List[PriceChange]:
        """Monitor recent price changes for optimization agent"""
        
    async def get_ownership_trends(self, days: int = 7) -> List[OwnershipChange]:
        """Get ownership percentage changes for rumor validation"""
```

## Data Refresh Strategy

### Coordination with fpl_data_collector

1. **Update Detection**
   - Monitor database file modification timestamps
   - Check for new gameweek data availability
   - Detect price changes and deadline updates

2. **Cache Invalidation**
   - Clear relevant caches when new data detected
   - Notify dependent agents of data updates
   - Implement graceful cache warm-up strategies

3. **Data Consistency**
   - Handle partial updates during data collection
   - Ensure atomic reads for multi-table queries
   - Validate data integrity after updates

### Real-time Data Needs

The Dugout agents require different data freshness levels:

- **Optimization Agent**: Real-time price changes and deadlines
- **Forecast Agent**: Updated after each gameweek completion  
- **Minutes Agent**: Updated with team news (daily)
- **Rumor Agent**: Requires external real-time feeds
- **Orchestrator**: Monitors all data freshness requirements

## Performance Considerations

### Database Optimization (Goal: predictable low-latency reads & freshness signaling)

1. **Indexing Strategy (Why)**
   ```sql
   -- Optimize common query patterns
   CREATE INDEX idx_players_team_position ON players(team_code, element_type);
   CREATE INDEX idx_gameweek_data_player_gw ON gameweek_data(player_id, gameweek);
   CREATE INDEX idx_fixtures_gameweek ON fixtures(gameweek);
   CREATE INDEX idx_price_changes_time ON price_changes(change_time);
   ```

2. **Query Patterns (Approach)**
   - Minimize N+1 query problems with efficient joins
   - Use prepared statements for repeated queries
   - Implement connection pooling for concurrent access

3. **Concurrent Access (Techniques)**
   - Read-heavy workload optimization
   - WAL mode for better concurrent read performance
   - Consider read replicas for analytical queries

### Monitoring and Metrics

- Query response time tracking (latency SLA)
- Connection pool utilization (capacity planning)
- Cache hit rates & invalidation efficiency (staleness avoidance)
- Data freshness & last-update deltas (proactive refresh triggers)
- Anomaly counts (quality alerts to Orchestrator)
- Error / retry rates (health & resilience)