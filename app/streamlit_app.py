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

from dugout.production.decisions import (
    get_captain_candidates,
    pick_captain,
    get_transfer_recommendations,
    optimize_free_hit,
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

TOP_N = 10

DISCLAIMER = """
⚠️ **Small differences (<0.5 pts) are noise.**  
This is decision support, not a guarantee.
"""

st.set_page_config(
    page_title="The Dugout",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_current_gw() -> int:
    """Infer next gameweek from available data."""
    from dugout.production.data.reader import DataReader

    reader = DataReader()
    raw_df = reader.get_all_gw_data()
    max_gw = int(raw_df["gw"].max())
    return min(max_gw + 1, 38)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.title("⚽ The Dugout")
st.sidebar.caption("Decision support for FPL managers")

page = st.sidebar.radio(
    "Decision",
    ["Captain", "Transfers", "Free Hit"],
    index=0,
)

st.sidebar.divider()

try:
    default_gw = get_current_gw()
except Exception:
    default_gw = 23

gw = st.sidebar.number_input(
    "Gameweek",
    min_value=1,
    max_value=38,
    value=default_gw,
)

st.sidebar.divider()
st.sidebar.caption("Decision Rule: `argmax(expected_points)`")
st.sidebar.caption("Research-validated. No heuristics.")

with st.sidebar.expander("ℹ️ How to use this"):
    st.markdown("""
    This tool helps you **choose between players**, not predict exact scores.
    
    - Picks are based on **expected points** for the upcoming gameweek
    - If two players are within **~0.5 pts**, treat them as equal
    - Use your own judgement for late injury news and rotation risk
    
    Designed to **reduce bad decisions over time**, not guarantee wins.
    """)

# -----------------------------------------------------------------------------
# Captain Page
# -----------------------------------------------------------------------------

def render_captain_page(gw: int):
    st.header("👑 Captain Recommendation")
    st.caption(f"GW{gw} · argmax(predicted_points)")
    st.info(DISCLAIMER)

    with st.spinner("Loading predictions..."):
        candidates, _, model_type = get_captain_candidates(gw=gw, top_n=TOP_N)

    if candidates.empty:
        st.warning("No available players found")
        return

    captain = pick_captain(candidates)

    col1, col2, col3 = st.columns([2, 1, 1])
    col1.metric("🎯 Captain", captain["player_name"], f"{captain['predicted_points']:.2f} pts")
    col2.metric("Team", captain["team_name"])
    col3.metric("Position", captain["position"])

    st.caption(f"Model: {model_type}")
    st.divider()

    st.subheader(f"Top {TOP_N} Candidates")

    display_df = candidates.copy()
    display_df["Rank"] = range(1, len(display_df) + 1)
    display_df["vs"] = display_df["opponent_short"]
    display_df["H/A"] = display_df["is_home"].map(lambda x: "H" if x else "A")

    st.dataframe(
        display_df.rename(columns={
            "player_name": "Player",
            "team_name": "Team",
            "position": "Pos",
            "predicted_points": "Expected Pts",
        })[["Rank", "Player", "Team", "vs", "H/A", "Pos", "Expected Pts"]],
        hide_index=True,
        use_container_width=True,
    )

# -----------------------------------------------------------------------------
# Transfers Page
# -----------------------------------------------------------------------------

def render_transfers_page(gw: int):
    st.header("⬆️ Transfer-In Recommendations")
    st.caption(f"GW{gw} · argmax(predicted_points)")
    st.info(DISCLAIMER)

    with st.spinner("Loading predictions..."):
        recs, _, model_type = get_transfer_recommendations(gw=gw, top_n=TOP_N)

    if recs.empty:
        st.warning("No available players found")
        return

    best = recs.iloc[0]

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    col1.metric("🎯 Top Transfer-In", best["player_name"], f"{best['predicted_points']:.2f} pts")
    col2.metric("Team", best["team_name"])
    col3.metric("Position", best["position"])
    col4.metric("Price", f"£{best['now_cost']:.1f}m")

    st.caption(f"Model: {model_type}")
    st.divider()

    st.subheader(f"Top {TOP_N} Recommendations")

    display_df = recs.copy()
    display_df["Rank"] = range(1, len(display_df) + 1)
    display_df["Price"] = display_df["now_cost"].map(lambda x: f"£{x:.1f}m")
    display_df["vs"] = display_df["opponent_short"]
    display_df["H/A"] = display_df["is_home"].map(lambda x: "H" if x else "A")

    st.dataframe(
        display_df.rename(columns={
            "player_name": "Player",
            "team_name": "Team",
            "position": "Pos",
            "predicted_points": "Expected Pts",
        })[["Rank", "Player", "Team", "vs", "H/A", "Pos", "Expected Pts", "Price"]],
        hide_index=True,
        use_container_width=True,
    )

# -----------------------------------------------------------------------------
# Free Hit Page
# -----------------------------------------------------------------------------

def render_free_hit_page(gw: int):
    st.header("🚀 Free Hit Optimizer")
    st.caption(f"GW{gw} · maximize Σ(predicted_points)")
    st.info(DISCLAIMER)

    with st.spinner("Optimizing squad..."):
        result, _, _, model_type = optimize_free_hit(gw=gw, budget=100.0)

    if result is None:
        st.error("Optimization failed")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Pts", f"{result.total_ev:.1f}")
    col2.metric("Formation", result.formation)
    col3.metric("Cost", f"£{result.total_cost:.1f}m")

    st.caption(f"Model: {model_type}")
    st.divider()

    st.subheader("Starting XI")

    xi_df = pd.DataFrame([
        {
            "Pos": p.get("pos"),
            "Player": p.get("name"),
            "Team": p.get("team"),
            "vs": p.get("opponent", ""),
            "H/A": "H" if p.get("is_home") else "A",
            "Cost": f"£{p.get('cost', 0):.1f}m",
            "Expected Pts": f"{p.get('ev', 0):.2f}",
        }
        for p in result.starting_xi
    ])

    st.dataframe(xi_df, hide_index=True, use_container_width=True)

    st.subheader("Bench")

    bench_df = pd.DataFrame([
        {
            "Order": i,
            "Role": "1st Sub" if i == 1 else "Fodder",
            "Pos": p.get("pos"),
            "Player": p.get("name"),
            "Cost": f"£{p.get('cost', 0):.1f}m",
        }
        for i, p in enumerate(result.bench, 1)
    ])

    st.dataframe(bench_df, hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if page == "Captain":
    render_captain_page(gw)
elif page == "Transfers":
    render_transfers_page(gw)
elif page == "Free Hit":
    render_free_hit_page(gw)

st.divider()
st.caption("The Dugout · argmax(predicted_points) · GitHub: gisaf22/the-dugout")
