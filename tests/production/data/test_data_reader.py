import os
import sqlite3

import pytest

from dugout.production.data import DataReader


# Use the current season database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fpl_2025_26.sqlite")


# Test: DataReader player lookup and recent history
@pytest.mark.skipif(not os.path.exists(DB_PATH), reason="Database not present")
def test_player_lookup_and_history():
    reader = DataReader(DB_PATH)

    # New API: get_available_players returns list of dicts for available players
    players = reader.get_available_players()[:5]
    assert len(players) > 0

    player = players[0]
    by_id = reader.get_player(player['id'])
    assert by_id is not None
    assert by_id['id'] == player['id']

    history = reader.get_player_history(player['id'], last_n=3)
    assert isinstance(history, list)
    assert len(history) <= 3
    for entry in history:
        assert entry['element'] == player['id']


# Test: DataReader fixtures and team lookup
@pytest.mark.skipif(not os.path.exists(DB_PATH), reason="Database not present")
def test_fixtures_and_team_lookup():
    reader = DataReader(DB_PATH)

    found = False
    for gw in range(1, 39):
        fixtures = reader.get_fixtures(gw=gw)
        if fixtures:
            found = True
            fixture = fixtures[0]
            team_id = fixture.get('team_h') or fixture.get('team_a')
            if team_id:
                team = reader.get_team(team_id)
                assert team is not None
            break
    assert found, "Expected at least one gameweek with fixtures"


# Test: DataReader current gameweek calculation
@pytest.mark.skipif(not os.path.exists(DB_PATH), reason="Database not present")
def test_get_current_gameweek():
    reader = DataReader(DB_PATH)
    gw = reader.get_current_gw()
    assert gw is None or gw >= 1


# Test: Bulk recent history with window-function fallback
def test_bulk_history(tmp_path):
    db_path = tmp_path / "sample.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE gameweeks (
                element_id INTEGER,
                round INTEGER,
                minutes INTEGER,
                total_points INTEGER,
                assists INTEGER,
                goals_scored INTEGER,
                bonus INTEGER,
                kickoff_time TEXT,
                fixture INTEGER
            );
            INSERT INTO gameweeks(element_id, round, minutes, total_points, kickoff_time, fixture) VALUES
                (1, 3, 90, 10, '2024-05-10', 1001),
                (1, 2, 80, 6, '2024-04-20', 1002),
                (1, 1, 75, 5, '2024-04-05', 1003),
                (2, 3, 60, 4, '2024-05-12', 2001),
                (2, 2, 30, 2, '2024-04-21', 2002);
            """
        )
        conn.commit()
    finally:
        conn.close()

    reader = DataReader(str(db_path))
    histories = reader.get_players_history_bulk([1, 2], last_n=2)
    assert len(histories) == 2
    assert len(histories[1]) == 2
    # Ordered by round ASC after filtering
    assert histories[1][0]['total_points'] == 6   # round 2
    assert histories[1][1]['total_points'] == 10  # round 3
    assert len(histories[2]) == 2
