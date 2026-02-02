"""The Dugout - FPL Decision Support Dashboard.

Three decisions, one rule: argmax(predicted_points).

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

TOP_N = 10
DB_PATH = Path(__file__).parent.parent / "storage" / "fpl_2025_26.sqlite"

st.set_page_config(
    page_title="The Dugout",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# DB Check (must happen before any imports that use DB)
# -----------------------------------------------------------------------------

def check_database() -> bool:
    """Check if SQLite database exists."""
    import os
    db_path = os.environ.get("DUGOUT_DB_PATH", str(DB_PATH))
    return Path(db_path).exists()


if not check_database():
    st.error("⚽ **Database not found**")
    st.markdown("""
    The FPL database hasn't been set up yet.
    
    **To fix this**, run the data pull script:
    ```
    PYTHONPATH=src python scripts/ops/pull_fpl_data.py
    ```
    
    Then refresh this page.
    """)
    st.stop()

# Now safe to import decision modules
from dugout.production.decisions import (
    get_captain_candidates,
    pick_captain,
    get_transfer_recommendations,
    optimize_free_hit,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_data_status() -> tuple[int, int]:
    """Return (current_gw, max_played_gw) for display."""
    from dugout.production.data.reader import DataReader

    reader = DataReader()
    
    # Get max GW with historical data
    raw_df = reader.get_all_gw_data()
    max_played_gw = int(raw_df["gw"].max())
    
    # Check if next GW has fixtures
    next_gw = min(max_played_gw + 1, 38)
    fixtures = reader.get_fixtures_for_gw(next_gw)
    
    if fixtures:
        return next_gw, max_played_gw
    else:
        return max_played_gw, max_played_gw


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.title("⚽ The Dugout")
st.sidebar.caption("Decision support for FPL managers")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "👑 Captain", "⬆️ Transfers", "🚀 Free Hit"],
    index=0,
)

try:
    current_gw, last_updated_gw = get_data_status()
except Exception:
    current_gw, last_updated_gw = 23, 23

# Only show GW selector on decision pages
if page != "🏠 Home":
    st.sidebar.divider()
    gw = st.sidebar.number_input(
        "Gameweek",
        min_value=1,
        max_value=38,
        value=current_gw,
    )
    st.sidebar.divider()
    st.sidebar.success(f"📅 Data updated through: **GW{last_updated_gw}**")
else:
    gw = current_gw  # default for routing

# -----------------------------------------------------------------------------
# Home Page
# -----------------------------------------------------------------------------

def render_home_page():
    st.title("⚽ Welcome to The Dugout")
    
    st.markdown("""
    **The Dugout** helps you make better Fantasy Premier League decisions.
    It uses a prediction model to estimate how many points each player will 
    score, then recommends the choice with the highest expected return.
    No gut feelings, no heuristics — just data.
    """)
    
    st.divider()
    
    st.subheader("📖 How to Use")
    
    st.markdown("""
    **👑 Captain** — Pick your captain for the week  
    Shows the player most likely to score the highest points. 
    Give them the armband.
    
    **⬆️ Transfers** — Find your next transfer target  
    Ranks all available players by expected points.
    Great for planning 1-week punts or long-term holds.
    
    **🚀 Free Hit** — Build a one-week dream team  
    Optimizes a full 15-player squad under budget.
    Use this when activating your Free Hit chip.
    """)
    
    st.divider()
    
    st.info("""
    ⚠️ **This is decision support, not a guarantee.**
    
    The model predicts *expected* points based on historical patterns.
    It can't predict injuries, rotation, or one-off performances.
    
    Use it to narrow down your choices — then trust your judgement.
    """)


# -----------------------------------------------------------------------------
# Captain Page
# -----------------------------------------------------------------------------

CAPTAIN_EXPLAINER = """
👑 **This is the player expected to score the most points this gameweek.**

"Expected Points" is our model's best estimate based on form, fixtures, 
and historical performance. Small differences (<0.5 pts) are noise — 
treat close calls as a toss-up.
"""

def render_captain_page(gw: int):
    st.header("👑 Captain Recommendation")
    st.caption(f"Gameweek {gw}")
    
    st.info(CAPTAIN_EXPLAINER)

    try:
        with st.spinner("Loading predictions..."):
            candidates, _, model_type = get_captain_candidates(gw=gw, top_n=TOP_N)
    except Exception as e:
        st.error(f"📅 **Data not available for GW{gw}**")
        st.markdown("This gameweek's fixtures haven't been released yet. Try a current or past gameweek.")
        return

    if candidates.empty:
        st.warning("No available players found for this gameweek.")
        return

    captain = pick_captain(candidates)

    st.success(f"### 🎯 Give the armband to: **{captain['player_name']}**")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Pts", f"{captain['predicted_points']:.1f}")
    col2.metric("Team", captain["team_name"])
    col3.metric("Position", captain["position"])
    col4.metric("Opponent", f"{captain.get('opponent_short', '?')} ({'H' if captain.get('is_home') else 'A'})")

    st.divider()

    st.subheader(f"Top {TOP_N} Captain Options")

    display_df = candidates.copy()
    display_df["Rank"] = range(1, len(display_df) + 1)
    display_df["Opponent"] = display_df.apply(
        lambda r: f"{r['opponent_short']} ({'H' if r['is_home'] else 'A'})", axis=1
    )

    st.dataframe(
        display_df.rename(columns={
            "player_name": "Player",
            "team_name": "Team",
            "position": "Pos",
            "predicted_points": "Expected Pts",
        })[["Rank", "Player", "Team", "Pos", "Opponent", "Expected Pts"]],
        hide_index=True,
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# Transfers Page
# -----------------------------------------------------------------------------

TRANSFER_EXPLAINER = """
⬆️ **Best players to bring in, ranked by expected points.**

This ignores ownership, form trends, and fixture swings — it's purely 
"who will score most this week?" Use it for short-term punts or 
to validate your transfer targets.
"""

def render_transfers_page(gw: int):
    st.header("⬆️ Transfer-In Recommendations")
    st.caption(f"Gameweek {gw}")
    
    st.info(TRANSFER_EXPLAINER)

    try:
        with st.spinner("Loading predictions..."):
            recs, _, model_type = get_transfer_recommendations(gw=gw, top_n=TOP_N)
    except Exception as e:
        st.error(f"📅 **Data not available for GW{gw}**")
        st.markdown("This gameweek's fixtures haven't been released yet. Try a current or past gameweek.")
        return

    if recs.empty:
        st.warning("No available players found for this gameweek.")
        return

    best = recs.iloc[0]

    st.success(f"### 🎯 Top pick: **{best['player_name']}** (£{best['now_cost']:.1f}m)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Pts", f"{best['predicted_points']:.1f}")
    col2.metric("Team", best["team_name"])
    col3.metric("Position", best["position"])
    col4.metric("Opponent", f"{best.get('opponent_short', '?')} ({'H' if best.get('is_home') else 'A'})")

    st.divider()

    st.subheader(f"Top {TOP_N} Transfer Targets")

    display_df = recs.copy()
    display_df["Rank"] = range(1, len(display_df) + 1)
    display_df["Price"] = display_df["now_cost"].map(lambda x: f"£{x:.1f}m")
    display_df["Opponent"] = display_df.apply(
        lambda r: f"{r['opponent_short']} ({'H' if r['is_home'] else 'A'})", axis=1
    )

    st.dataframe(
        display_df.rename(columns={
            "player_name": "Player",
            "team_name": "Team",
            "position": "Pos",
            "predicted_points": "Expected Pts",
        })[["Rank", "Player", "Team", "Pos", "Opponent", "Expected Pts", "Price"]],
        hide_index=True,
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# Free Hit Page
# -----------------------------------------------------------------------------

FREE_HIT_EXPLAINER = """
🚀 **This squad maximizes total expected points for one gameweek.**

The optimizer picks the best 15 players under £100m budget, 
following FPL rules (max 3 per team, valid formation). 
Use this when activating your Free Hit chip.
"""

POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

def render_free_hit_page(gw: int):
    st.header("🚀 Free Hit Optimizer")
    st.caption(f"Gameweek {gw}")
    
    st.info(FREE_HIT_EXPLAINER)

    try:
        with st.spinner("Optimizing squad (this may take a few seconds)..."):
            result, _, _, model_type = optimize_free_hit(gw=gw, budget=100.0)
    except Exception as e:
        st.error(f"📅 **Data not available for GW{gw}**")
        st.markdown("This gameweek's fixtures haven't been released yet. Try a current or past gameweek.")
        return

    if result is None:
        st.error("Optimization failed — try a different gameweek.")
        return

    # Summary metrics
    st.success(f"### ⚽ Optimal Squad Found")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Expected Pts", f"{result.total_ev:.1f}")
    col2.metric("Formation", f"**{result.formation}**")
    col3.metric("Budget Used", f"£{result.total_cost:.1f}m / £100m")

    st.divider()

    # Starting XI - grouped by position
    st.subheader("Starting XI")
    
    xi_data = []
    for p in result.starting_xi:
        pos = p.get("pos", "?")
        xi_data.append({
            "_pos_order": POS_ORDER.get(pos, 9),
            "Pos": pos,
            "Player": p.get("name") or p.get("player_name"),
            "Team": p.get("team") or p.get("team_name"),
            "Opponent": f"{p.get('opponent_short', '?')} ({'H' if p.get('is_home') else 'A'})",
            "Cost": f"£{p.get('cost', 0) or p.get('now_cost', 0):.1f}m",
            "Expected Pts": f"{p.get('ev', 0) or p.get('predicted_points', 0):.1f}",
        })
    
    xi_df = pd.DataFrame(xi_data).sort_values("_pos_order").drop(columns=["_pos_order"])
    
    st.dataframe(xi_df, hide_index=True, use_container_width=True)

    # Bench
    st.subheader("Bench")
    st.caption("Bench order: 1st sub plays if a starter doesn't. Fodder are budget fillers.")

    bench_data = []
    for i, p in enumerate(result.bench, 1):
        bench_data.append({
            "#": i,
            "Role": "1st Sub" if i == 1 else "Fodder",
            "Pos": p.get("pos"),
            "Player": p.get("name") or p.get("player_name"),
            "Cost": f"£{p.get('cost', 0) or p.get('now_cost', 0):.1f}m",
        })

    bench_df = pd.DataFrame(bench_data)
    st.dataframe(bench_df, hide_index=True, use_container_width=True)


# -----------------------------------------------------------------------------
# Main Router
# -----------------------------------------------------------------------------

if page == "🏠 Home":
    render_home_page()
elif page == "👑 Captain":
    render_captain_page(gw)
elif page == "⬆️ Transfers":
    render_transfers_page(gw)
elif page == "🚀 Free Hit":
    render_free_hit_page(gw)

# Footer
st.divider()
st.caption("⚽ The Dugout · Decision support for FPL · [GitHub](https://github.com/gisaf22/the-dugout)")
