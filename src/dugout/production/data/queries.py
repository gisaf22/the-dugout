"""Centralized SQL queries for FPL database access.

All SQL statements used by DataReader and scripts live here.
Named constants for clarity and single source of truth.
"""

# -----------------------------------------------------------------------------
# Player queries
# -----------------------------------------------------------------------------

PLAYER_BY_ID = """
SELECT id, first_name, second_name, web_name, team, element_type,
       now_cost, status, chance_of_playing_next_round
FROM players WHERE id = ?
"""

PLAYERS_BASE = "SELECT * FROM players WHERE 1=1"

PLAYERS_AVAILABLE = """
SELECT * FROM players 
WHERE status NOT IN ('i', 'n', 's', 'u')
ORDER BY total_points DESC
"""

# -----------------------------------------------------------------------------
# Team queries
# -----------------------------------------------------------------------------

TEAM_BY_ID = "SELECT * FROM teams WHERE id = ?"

TEAMS_ALL = "SELECT * FROM teams ORDER BY id"

# -----------------------------------------------------------------------------
# Fixture queries
# -----------------------------------------------------------------------------

FIXTURES_ALL = "SELECT * FROM fixtures ORDER BY event, kickoff_time"

FIXTURES_BY_GW = "SELECT * FROM fixtures WHERE event = ? ORDER BY kickoff_time"

FIXTURES_NEXT_FOR_TEAM = """
SELECT * FROM fixtures 
WHERE (team_h = ? OR team_a = ?) AND finished = 0
ORDER BY event LIMIT ?
"""

FIXTURES_FINISHED = """
SELECT event as gw, team_h, team_a, team_h_score, team_a_score
FROM fixtures WHERE finished = 1
"""

# -----------------------------------------------------------------------------
# Gameweek queries
# -----------------------------------------------------------------------------

PLAYER_HISTORY = """
SELECT * FROM gameweeks 
WHERE element_id = ?
ORDER BY round DESC
LIMIT ?
"""

CURRENT_GW = "SELECT MAX(round) as gw FROM gameweeks"

GW_DATA_SINGLE = """
SELECT g.*, p.web_name, p.now_cost, p.status, t.short_name as team_name
FROM gameweeks g
JOIN players p ON g.element_id = p.id
JOIN teams t ON p.team = t.id
WHERE g.round = ?
"""

# -----------------------------------------------------------------------------
# Bulk queries (feature engineering)
# -----------------------------------------------------------------------------

# Placeholder count must be filled: .format(",".join("?" for _ in ids))
PLAYERS_HISTORY_BULK = """
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY element_id ORDER BY round DESC) AS rn
    FROM gameweeks
    WHERE element_id IN ({placeholders})
)
SELECT * FROM ranked WHERE rn <= ?
ORDER BY element_id, round
"""

# -----------------------------------------------------------------------------
# Full gameweek data (for training/analysis)
# -----------------------------------------------------------------------------

GW_DATA_FULL = """
SELECT 
    g.element_id as player_id,
    p.web_name as player_name,
    p.first_name,
    p.second_name,
    t.name as team_name,
    p.team as team_id,
    p.element_type as position,
    p.now_cost,
    p.status,
    g.round as gw,
    g.total_points,
    g.minutes,
    g.goals_scored,
    g.assists,
    g.clean_sheets,
    g.goals_conceded,
    g.bonus,
    g.bps,
    g.ict_index,
    g.influence,
    g.creativity,
    g.threat,
    g.expected_goals as xG,
    g.expected_assists as xA,
    g.expected_goal_involvements as xGI,
    g.expected_goals_conceded as xGC,
    g.starts,
    -- Fixture data
    CASE WHEN f.team_h = p.team THEN 1 ELSE 0 END as is_home,
    CASE WHEN f.team_h = p.team THEN f.team_a ELSE f.team_h END as opponent_id,
    CASE WHEN f.team_h = p.team THEN opp.name ELSE opp_h.name END as opponent_name,
    CASE WHEN f.team_h = p.team THEN opp.short_name ELSE opp_h.short_name END as opponent_short,
    -- FDR approximation from team strengths
    CASE 
        WHEN f.team_h = p.team THEN opp.strength_overall_away
        ELSE opp_h.strength_overall_home
    END as opponent_strength,
    f.team_h_score as fixture_home_goals,
    f.team_a_score as fixture_away_goals,
    f.finished as fixture_finished
FROM gameweeks g
JOIN players p ON g.element_id = p.id
JOIN teams t ON p.team = t.id
LEFT JOIN fixtures f ON f.event = g.round 
    AND (f.team_h = p.team OR f.team_a = p.team)
LEFT JOIN teams opp ON f.team_a = opp.id
LEFT JOIN teams opp_h ON f.team_h = opp_h.id
ORDER BY g.element_id, g.round
"""

# Alias for backwards compatibility
TRAINING_DATA_BASE = GW_DATA_FULL
