"""
Stage 8a — Multi-GW Belief Rollup

Aggregates single-GW beliefs into multi-GW rollups to prepare for
horizon-based decision evaluation in Stage 8b.

This is a pure belief transformation, not a modeling or decision stage.

Motivation:
    Single-GW decisions showed that availability (p_play) does not improve
    ranking outcomes. Stage 8a tests whether availability compounds over
    time when decisions persist across multiple gameweeks.

Inputs:
    storage/datasets/beliefs.csv

Outputs:
    storage/datasets/beliefs_multigw.csv

Schema:
    player_id       int     Player identifier
    gw_start        int     Starting gameweek (t)
    horizon         int     Number of future GWs (H)
    cum_mu_points   float   Σ mu_points over horizon
    cum_ev          float   Σ (p_play × mu_points) over horizon
    cum_play_prob   float   Π p_play over horizon
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

HORIZONS = [2, 3, 4, 5]


# -----------------------------------------------------------------------------
# Rollup Computation
# -----------------------------------------------------------------------------


def compute_multigw_rollup(
    beliefs: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Compute multi-GW belief rollup for a fixed horizon.
    
    For each (player_id, gw_start), aggregates beliefs over GWs:
        gw_start, gw_start+1, ..., gw_start+horizon-1
    
    Only includes rows where all H future GWs exist.
    
    Returns:
        DataFrame with player_id, gw_start, horizon, cum_mu_points, cum_ev, cum_play_prob
    """
    # Get max GW to determine valid starting points
    max_gw = beliefs["gw"].max()
    
    # Valid gw_start values: must have horizon GWs available
    valid_gw_starts = [gw for gw in beliefs["gw"].unique() if gw + horizon - 1 <= max_gw]
    
    if not valid_gw_starts:
        return pd.DataFrame(columns=[
            "player_id", "gw_start", "horizon", "cum_mu_points", "cum_ev", "cum_play_prob"
        ])
    
    # Build rollup for each valid gw_start
    rollups = []
    
    for gw_start in valid_gw_starts:
        # Get GWs in this horizon
        gw_range = list(range(gw_start, gw_start + horizon))
        
        # Filter beliefs to this GW range
        horizon_beliefs = beliefs[beliefs["gw"].isin(gw_range)].copy()
        
        # Group by player and aggregate
        player_rollup = horizon_beliefs.groupby("player_id").agg(
            n_gws=("gw", "count"),
            cum_mu_points=("mu_points", "sum"),
            # For cum_ev, we need p_play × mu_points per row first
            cum_play_prob=("p_play", "prod"),
        ).reset_index()
        
        # Compute cum_ev separately (sum of p_play × mu_points)
        horizon_beliefs["ev"] = horizon_beliefs["p_play"].fillna(0) * horizon_beliefs["mu_points"].fillna(0)
        ev_sum = horizon_beliefs.groupby("player_id")["ev"].sum().reset_index()
        ev_sum.columns = ["player_id", "cum_ev"]
        
        player_rollup = player_rollup.merge(ev_sum, on="player_id", how="left")
        
        # Only keep players with complete horizon data
        player_rollup = player_rollup[player_rollup["n_gws"] == horizon].copy()
        
        # Add metadata
        player_rollup["gw_start"] = gw_start
        player_rollup["horizon"] = horizon
        
        rollups.append(player_rollup[[
            "player_id", "gw_start", "horizon", 
            "cum_mu_points", "cum_ev", "cum_play_prob"
        ]])
    
    return pd.concat(rollups, ignore_index=True)


def build_multigw_beliefs(beliefs: pd.DataFrame) -> pd.DataFrame:
    """
    Build multi-GW belief rollups for all horizons.
    
    Horizons: H ∈ {2, 3, 4, 5}
    """
    all_rollups = []
    
    for horizon in HORIZONS:
        rollup = compute_multigw_rollup(beliefs, horizon)
        all_rollups.append(rollup)
    
    return pd.concat(all_rollups, ignore_index=True)


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    """Stage 8a pipeline entry point."""
    project_root = Path(__file__).resolve().parents[4]
    
    print("=" * 70)
    print("Stage 8a — Multi-GW Belief Rollup")
    print("=" * 70)
    
    # Load beliefs
    print("\n1. Loading beliefs...")
    beliefs_path = project_root / "storage" / "research" / "datasets" / "beliefs.csv"
    beliefs = pd.read_csv(beliefs_path)
    
    print(f"   Rows: {len(beliefs):,}")
    print(f"   Players: {beliefs['player_id'].nunique():,}")
    print(f"   GWs: {beliefs['gw'].min()} to {beliefs['gw'].max()}")
    
    # Compute rollups
    print("\n2. Computing multi-GW rollups...")
    rollups = build_multigw_beliefs(beliefs)
    
    print(f"   Total rollup rows: {len(rollups):,}")
    
    # Summary by horizon
    print("\n   Rows by horizon:")
    for h in HORIZONS:
        h_rows = len(rollups[rollups["horizon"] == h])
        print(f"     H={h}: {h_rows:,} rows")
    
    # Save output
    output_path = project_root / "storage" / "research" / "datasets" / "beliefs_multigw.csv"
    rollups.to_csv(output_path, index=False)
    print(f"\n3. Saved: {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Multi-GW Rollup Summary")
    print("=" * 70)
    
    print(f"\n{'Horizon':<10} {'Avg cum_mu_points':>20} {'Avg cum_ev':>15} {'Avg cum_play_prob':>20}")
    print("-" * 70)
    
    for h in HORIZONS:
        h_data = rollups[rollups["horizon"] == h]
        avg_mu = h_data["cum_mu_points"].mean()
        avg_ev = h_data["cum_ev"].mean()
        avg_prob = h_data["cum_play_prob"].mean()
        print(f"H={h:<7} {avg_mu:>20.2f} {avg_ev:>15.2f} {avg_prob:>20.3f}")
    
    # Show availability decay
    print("\n" + "-" * 70)
    print("Availability Decay (avg cum_play_prob):")
    for h in HORIZONS:
        h_data = rollups[rollups["horizon"] == h]
        avg_prob = h_data["cum_play_prob"].mean()
        print(f"  H={h}: {avg_prob:.3f} (≈ 0.95^{h} = {0.95**h:.3f})")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
