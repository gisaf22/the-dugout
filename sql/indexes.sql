-- Players indexes
CREATE INDEX IF NOT EXISTS idx_players_id ON players(id);
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team);
CREATE INDEX IF NOT EXISTS idx_players_element_type ON players(element_type);

-- Gameweeks indexes
CREATE INDEX IF NOT EXISTS idx_gw_element ON gameweeks(element);
CREATE INDEX IF NOT EXISTS idx_gw_fixture ON gameweeks(fixture);
CREATE INDEX IF NOT EXISTS idx_gw_kickoff_time ON gameweeks(kickoff_time);

-- Fixtures indexes
CREATE INDEX IF NOT EXISTS idx_fixtures_id ON fixtures(id);
CREATE INDEX IF NOT EXISTS idx_fixtures_gameweek ON fixtures(gameweek);
CREATE INDEX IF NOT EXISTS idx_fixtures_kickoff_time ON fixtures(kickoff_time);
CREATE INDEX IF NOT EXISTS idx_fixtures_teams ON fixtures(team_h, team_a);

-- Teams index
CREATE INDEX IF NOT EXISTS idx_teams_id ON teams(id);
