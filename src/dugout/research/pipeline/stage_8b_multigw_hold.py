"""
Stage 8b — Multi-GW Hold Decision Evaluation

Evaluates multi-GW hold decisions using belief rollups from Stage 8a.

Hypothesis:
    Availability compounds over time. Therefore, for multi-GW holds:
    cum_ev = Σ (p_play × mu_points) should outperform cum_mu_points = Σ mu_points
    in terms of regret, especially at longer horizons.

Policies:
    A (cum_mu): Select player with highest cum_mu_points
    B (cum_ev): Select player with highest cum_ev

Inputs:
    storage/datasets/beliefs_multigw.csv
    storage/datasets/targets.csv

Outputs:
    storage/datasets/evaluation_multigw_hold.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def load_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load multi-GW beliefs and targets."""
    rollups = pd.read_csv(project_root / "storage" / "research" / "datasets" / "beliefs_multigw.csv")
    targets = pd.read_csv(project_root / "storage" / "research" / "datasets" / "targets.csv")
    return rollups, targets


# -----------------------------------------------------------------------------
# Oracle Computation
# -----------------------------------------------------------------------------


def compute_realized_points(
    targets: pd.DataFrame,
    gw_start: int,
    horizon: int,
) -> pd.DataFrame:
    """
    Compute realized total points for each player over a horizon.
    
    Returns: player_id, total_points
    """
    gw_range = list(range(gw_start, gw_start + horizon))
    
    horizon_targets = targets[targets["gw"].isin(gw_range)].copy()
    
    # Sum points per player
    realized = horizon_targets.groupby("player_id")["y_points"].sum().reset_index()
    realized.columns = ["player_id", "total_points"]
    
    # Only keep players with complete horizon data
    gw_counts = horizon_targets.groupby("player_id")["gw"].nunique().reset_index()
    gw_counts.columns = ["player_id", "n_gws"]
    
    realized = realized.merge(gw_counts, on="player_id")
    realized = realized[realized["n_gws"] == horizon][["player_id", "total_points"]]
    
    return realized


# -----------------------------------------------------------------------------
# Policy Evaluation
# -----------------------------------------------------------------------------


def evaluate_policy(
    rollups: pd.DataFrame,
    targets: pd.DataFrame,
    score_col: str,
    policy_name: str,
) -> pd.DataFrame:
    """
    Evaluate a single policy across all (gw_start, horizon) combinations.
    
    Args:
        rollups: Multi-GW belief rollups
        targets: Target data
        score_col: Column to use for ranking (cum_mu_points or cum_ev)
        policy_name: Name for output
    
    Returns:
        Evaluation DataFrame with regret per decision
    """
    results = []
    
    for (gw_start, horizon), group in rollups.groupby(["gw_start", "horizon"]):
        # Select player with highest score
        idx_max = group[score_col].idxmax()
        chosen_player = group.loc[idx_max, "player_id"]
        
        # Get realized points for all players
        realized = compute_realized_points(targets, gw_start, horizon)
        
        if len(realized) == 0:
            continue
        
        # Get chosen player's realized points
        chosen_row = realized[realized["player_id"] == chosen_player]
        if len(chosen_row) == 0:
            chosen_total = 0
        else:
            chosen_total = chosen_row["total_points"].values[0]
        
        # Get oracle (best realized)
        optimal_idx = realized["total_points"].idxmax()
        optimal_player = realized.loc[optimal_idx, "player_id"]
        optimal_total = realized.loc[optimal_idx, "total_points"]
        
        # Compute regret
        regret = optimal_total - chosen_total
        
        results.append({
            "gw_start": gw_start,
            "horizon": horizon,
            "policy": policy_name,
            "chosen_player_id": chosen_player,
            "chosen_total_points": chosen_total,
            "optimal_player_id": optimal_player,
            "optimal_total_points": optimal_total,
            "regret": regret,
        })
    
    return pd.DataFrame(results)


def compute_metrics(evaluation: pd.DataFrame, horizon: int) -> dict:
    """Compute aggregate regret metrics for a policy at a given horizon."""
    df = evaluation[evaluation["horizon"] == horizon]
    regret = df["regret"]
    
    return {
        "mean_regret": regret.mean(),
        "median_regret": regret.median(),
        "pct_high_regret": (regret >= 10).mean(),
        "total_regret": regret.sum(),
        "n_decisions": len(regret),
    }


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    """Stage 8b pipeline entry point."""
    project_root = Path(__file__).resolve().parents[4]
    
    print("=" * 70)
    print("Stage 8b — Multi-GW Hold Decision Evaluation")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    rollups, targets = load_data(project_root)
    
    print(f"   Rollups: {len(rollups):,} rows")
    print(f"   Targets: {len(targets):,} rows")
    print(f"   Horizons: {sorted(rollups['horizon'].unique())}")
    
    # Evaluate policies
    print("\n2. Evaluating policies...")
    
    eval_cum_mu = evaluate_policy(rollups, targets, "cum_mu_points", "cum_mu")
    eval_cum_ev = evaluate_policy(rollups, targets, "cum_ev", "cum_ev")
    
    # Combine
    evaluation = pd.concat([eval_cum_mu, eval_cum_ev], ignore_index=True)
    
    print(f"   Total evaluations: {len(evaluation):,}")
    
    # Save output
    output_path = project_root / "storage" / "research" / "datasets" / "evaluation_multigw_hold.csv"
    evaluation.to_csv(output_path, index=False)
    print(f"\n3. Saved: {output_path}")
    
    # Summary by horizon and policy
    print("\n" + "=" * 70)
    print("Multi-GW Hold Decision Results")
    print("=" * 70)
    
    horizons = sorted(rollups["horizon"].unique())
    
    print(f"\n{'Horizon':<10} {'Policy':<12} {'Mean Regret':>12} {'Median':>10} {'% ≥10':>10}")
    print("-" * 60)
    
    for h in horizons:
        for policy in ["cum_mu", "cum_ev"]:
            df = evaluation[(evaluation["horizon"] == h) & (evaluation["policy"] == policy)]
            if len(df) == 0:
                continue
            
            mean_r = df["regret"].mean()
            median_r = df["regret"].median()
            pct_high = (df["regret"] >= 10).mean()
            
            print(f"H={h:<7} {policy:<12} {mean_r:>12.2f} {median_r:>10.1f} {pct_high:>10.1%}")
    
    # Compare policies at each horizon
    print("\n" + "-" * 60)
    print("Policy Comparison (cum_ev vs cum_mu):")
    print("-" * 60)
    
    for h in horizons:
        cum_mu_df = evaluation[(evaluation["horizon"] == h) & (evaluation["policy"] == "cum_mu")]
        cum_ev_df = evaluation[(evaluation["horizon"] == h) & (evaluation["policy"] == "cum_ev")]
        
        if len(cum_mu_df) == 0 or len(cum_ev_df) == 0:
            continue
        
        delta = cum_ev_df["regret"].mean() - cum_mu_df["regret"].mean()
        
        if delta < 0:
            result = f"✅ cum_ev wins by {abs(delta):.2f}"
        elif delta > 0:
            result = f"❌ cum_mu wins by {delta:.2f}"
        else:
            result = "➖ tie"
        
        print(f"  H={h}: {result}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
