"""The Dugout - FPL Decision Support Dashboard.

Minimal Streamlit MVP: Captain, Transfers, Free Hit.
Uses frozen decision functions from dugout.production.decisions.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="The Dugout",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import decision functions
from dugout.production.decisions import (
    get_captain_candidates,
    pick_captain,
    get_transfer_recommendations,
    optimize_free_hit,
)


def get_current_gw() -> int:
    """Get the next gameweek to predict."""
    from dugout.production.data.reader import DataReader
    reader = DataReader()
    raw_df = reader.get_all_gw_data()
    return int(raw_df["gw"].max()) + 1


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("⚽ The Dugout")
st.sidebar.caption("Decision support for FPL managers")

page = st.sidebar.radio(
    "Decision",
    ["Captain", "Transfers", "Free Hit"],
    index=0,
)

st.sidebar.divider()

# Gameweek selector
try:
    default_gw = get_current_gw()
except Exception:
    default_gw = 23

gw = st.sidebar.number_input("Gameweek", min_value=1, max_value=38, value=default_gw)

st.sidebar.divider()
st.sidebar.caption("Decision Rule: `argmax(expected_points)`")
st.sidebar.caption("Research-validated. No heuristics.")


# =============================================================================
# Captain Page
# =============================================================================

def render_captain_page(gw: int):
    st.header("👑 Captain Recommendation")
    st.caption(f"GW{gw} · Decision: argmax(predicted_points)")
    
    with st.spinner("Loading predictions..."):
        try:
            candidates, target_gw = get_captain_candidates(gw=gw, top_n=10)
        except Exception as e:
            st.error(f"Error: {e}")
            return
    
    if candidates.empty:
        st.warning("No available players found")
        return
    
    # Pick captain
    captain = pick_captain(candidates)
    
    # Hero card
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.metric(
            label="🎯 Captain",
            value=captain["player_name"],
            delta=f"{captain['predicted_points']:.2f} pts",
        )
    with col2:
        st.metric("Team", captain["team_name"])
    with col3:
        st.metric("Position", captain["position"])
    
    st.divider()
    
    # Top candidates table
    st.subheader("Top 10 Candidates")
    
    display_df = candidates.copy()
    display_df["Rank"] = range(1, len(display_df) + 1)
    display_df = display_df.rename(columns={
        "player_name": "Player",
        "team_name": "Team",
        "position": "Pos",
        "predicted_points": "Expected Pts",
    })
    
    st.dataframe(
        display_df[["Rank", "Player", "Team", "Pos", "Expected Pts"]],
        hide_index=True,
        use_container_width=True,
    )
    
    # Guidance
    with st.expander("📋 Interpretation Guidance"):
        st.markdown("""
        - **Small differences (<0.5 pts) = noise** — don't overthink
        - **Override only for**: confirmed injury, suspension, verified news
        - This is **decision-support**, not a guarantee
        """)


# =============================================================================
# Transfers Page
# =============================================================================

def render_transfers_page(gw: int):
    st.header("⬆️ Transfer-In Recommendations")
    st.caption(f"GW{gw} · Decision: argmax(predicted_points)")
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Number of recommendations", 5, 20, 10)
    
    with st.spinner("Loading predictions..."):
        try:
            recommendations, target_gw = get_transfer_recommendations(
                gw=gw, top_n=top_n
            )
        except Exception as e:
            st.error(f"Error: {e}")
            return
    
    if recommendations.empty:
        st.warning("No available players found")
        return
    
    # Best pick
    best = recommendations.iloc[0]
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.metric(
            label="🎯 Top Transfer-In",
            value=best["player_name"],
            delta=f"{best['predicted_points']:.2f} pts",
        )
    with col2:
        st.metric("Team", best["team_name"])
    with col3:
        st.metric("Position", best["position"])
    with col4:
        st.metric("Price", f"£{best['now_cost']:.1f}m")
    
    st.divider()
    
    # Recommendations table
    st.subheader(f"Top {top_n} Recommendations")
    
    display_df = recommendations.copy()
    display_df["Rank"] = range(1, len(display_df) + 1)
    display_df["Price"] = display_df["now_cost"].apply(lambda x: f"£{x:.1f}m")
    display_df = display_df.rename(columns={
        "player_name": "Player",
        "team_name": "Team",
        "position": "Pos",
        "predicted_points": "Expected Pts",
    })
    
    st.dataframe(
        display_df[["Rank", "Player", "Team", "Pos", "Expected Pts", "Price"]],
        hide_index=True,
        use_container_width=True,
    )


# =============================================================================
# Free Hit Page
# =============================================================================

def render_free_hit_page(gw: int):
    st.header("🚀 Free Hit Optimizer")
    st.caption(f"GW{gw} · Maximize Σ(predicted_points)")
    
    # Budget slider
    budget = st.slider("Budget (£m)", 95.0, 105.0, 100.0, 0.5)
    
    with st.spinner("Optimizing squad..."):
        try:
            result, df, target_gw = optimize_free_hit(gw=gw, budget=budget)
        except Exception as e:
            st.error(f"Error: {e}")
            return
    
    if result is None:
        st.error("Optimization failed")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total EV", f"{result.total_ev:.1f} pts")
    col2.metric("Formation", result.formation)
    col3.metric("Cost", f"£{result.total_cost:.1f}m")
    col4.metric("ITB", f"£{budget - result.total_cost:.1f}m")
    
    st.divider()
    
    # Captain/Vice
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**👑 Captain:** {result.captain['name']}")
    with col2:
        st.markdown(f"**🥈 Vice:** {result.vice_captain['name']}")
    
    st.divider()
    
    # Starting XI
    st.subheader("Starting XI")
    
    xi_data = []
    for p in result.starting_xi:
        xi_data.append({
            "Pos": p.get("pos", ""),
            "Player": p.get("name", ""),
            "Team": p.get("team", ""),
            "vs": p.get("fixture_opponent", ""),
            "H/A": "H" if p.get("fixture_home") else "A",
            "Cost": f"£{p.get('cost', 0):.1f}m",
            "EV": f"{p.get('ev', 0):.2f}",
        })
    
    st.dataframe(pd.DataFrame(xi_data), hide_index=True, use_container_width=True)
    
    # Bench
    st.subheader("Bench")
    
    bench_data = []
    for i, p in enumerate(result.bench, 1):
        role = "1st Sub" if i == 1 else "Fodder"
        bench_data.append({
            "Order": i,
            "Role": role,
            "Pos": p.get("pos", ""),
            "Player": p.get("name", ""),
            "Cost": f"£{p.get('cost', 0):.1f}m",
        })
    
    st.dataframe(pd.DataFrame(bench_data), hide_index=True, use_container_width=True)


# =============================================================================
# Main
# =============================================================================

if page == "Captain":
    render_captain_page(gw)
elif page == "Transfers":
    render_transfers_page(gw)
elif page == "Free Hit":
    render_free_hit_page(gw)

# Footer
st.divider()
st.caption("The Dugout · Decision support, not automation · [Decision Contract](docs/DECISION_CONTRACT_LAYER.md)")
