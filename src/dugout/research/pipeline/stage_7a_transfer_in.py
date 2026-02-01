"""
Stage 7a — Transfer-IN Ranking (Single-GW)

Evaluates whether availability-adjusted expected value improves transfer-IN
decisions, in contrast to captaincy where it failed.

Hypothesis:
    For transfer-IN decisions, minimizing the risk of zero minutes matters.
    Therefore: score = p_play × mu_points should outperform mu_points alone.

Policy:
    For each GW, select the player with highest score = p_play × mu_points.

Baselines:
    A: mu_points only
    B: random player
    C: points_per_90_5 (historical PPG)

Outputs:
    storage/datasets/evaluation_transfer_in.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Policy Functions
# -----------------------------------------------------------------------------


def policy_availability_adjusted(beliefs: pd.DataFrame) -> pd.DataFrame:
    """
    Transfer-IN policy: p_play × mu_points.
    
    Returns player_id with highest score per GW.
    """
    df = beliefs[["player_id", "gw", "p_play", "mu_points"]].copy()
    df["score"] = df["p_play"].fillna(0) * df["mu_points"].fillna(0)
    
    idx_max = df.groupby("gw")["score"].idxmax()
    return df.loc[idx_max, ["gw", "player_id"]].rename(
        columns={"player_id": "chosen_player_id"}
    )


def policy_mu_points_only(beliefs: pd.DataFrame) -> pd.DataFrame:
    """Baseline A: mu_points only."""
    df = beliefs[["player_id", "gw", "mu_points"]].copy()
    df["score"] = df["mu_points"].fillna(0)
    
    idx_max = df.groupby("gw")["score"].idxmax()
    return df.loc[idx_max, ["gw", "player_id"]].rename(
        columns={"player_id": "chosen_player_id"}
    )


def policy_random(beliefs: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Baseline B: random player per GW."""
    np.random.seed(seed)
    
    choices = []
    for gw, group in beliefs.groupby("gw"):
        idx = np.random.choice(group.index)
        choices.append({"gw": gw, "chosen_player_id": group.loc[idx, "player_id"]})
    
    return pd.DataFrame(choices)


def policy_points_per_90(beliefs: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    """Baseline C: points_per_90_5 from performance features."""
    df = beliefs[["player_id", "gw"]].merge(
        performance[["player_id", "gw", "points_per_90_5"]],
        on=["player_id", "gw"],
        how="left"
    )
    df["score"] = df["points_per_90_5"].fillna(0)
    
    idx_max = df.groupby("gw")["score"].idxmax()
    return df.loc[idx_max, ["gw", "player_id"]].rename(
        columns={"player_id": "chosen_player_id"}
    )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def evaluate_policy(
    choices: pd.DataFrame,
    targets: pd.DataFrame,
    policy_name: str,
) -> pd.DataFrame:
    """
    Evaluate transfer-IN choices using regret.
    
    Returns evaluation table with regret per GW.
    """
    # Get chosen player's actual points
    evaluation = choices.merge(
        targets[["player_id", "gw", "y_points"]].rename(
            columns={"player_id": "chosen_player_id", "y_points": "chosen_points"}
        ),
        on=["gw", "chosen_player_id"],
        how="left",
    )
    
    # Find optimal (max points among players who played)
    targets_played = targets[targets["y_play"] == 1]
    optimal = (
        targets_played.loc[targets_played.groupby("gw")["y_points"].idxmax()]
        [["gw", "player_id", "y_points"]]
        .rename(columns={"player_id": "optimal_player_id", "y_points": "optimal_points"})
    )
    
    # Merge
    evaluation = evaluation.merge(optimal, on="gw", how="left")
    
    # Handle missing (chosen player didn't play)
    evaluation["chosen_points"] = evaluation["chosen_points"].fillna(0).astype(int)
    evaluation["optimal_points"] = evaluation["optimal_points"].fillna(0).astype(int)
    
    # Compute regret
    evaluation["regret"] = evaluation["optimal_points"] - evaluation["chosen_points"]
    evaluation["policy_name"] = policy_name
    
    return evaluation[[
        "policy_name", "gw", "chosen_player_id", "chosen_points",
        "optimal_player_id", "optimal_points", "regret"
    ]]


def compute_metrics(evaluation: pd.DataFrame) -> dict:
    """Compute aggregate regret metrics for a policy."""
    regret = evaluation["regret"]
    return {
        "mean_regret": regret.mean(),
        "median_regret": regret.median(),
        "pct_high_regret": (regret >= 10).mean(),
        "total_regret": regret.sum(),
        "n_gw": len(regret),
    }


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    """Stage 7a pipeline entry point."""
    project_root = Path(__file__).resolve().parents[4]
    
    print("=" * 70)
    print("Stage 7a — Transfer-IN Ranking (Single-GW)")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    beliefs = pd.read_csv(project_root / "storage" / "research" / "datasets" / "beliefs.csv")
    targets = pd.read_csv(project_root / "storage" / "research" / "datasets" / "targets.csv")
    performance = pd.read_csv(project_root / "storage" / "research" / "datasets" / "features_performance.csv")
    
    print(f"   Beliefs: {len(beliefs):,} rows")
    print(f"   Targets: {len(targets):,} rows")
    print(f"   GWs: {beliefs['gw'].min()} to {beliefs['gw'].max()}")
    
    # Run policies
    print("\n2. Running policies...")
    
    policies = {
        "stage_7a_p_play_x_mu_points": policy_availability_adjusted(beliefs),
        "baseline_a_mu_points": policy_mu_points_only(beliefs),
        "baseline_b_random": policy_random(beliefs),
        "baseline_c_points_per_90": policy_points_per_90(beliefs, performance),
    }
    
    # Evaluate each policy
    print("\n3. Evaluating policies...")
    
    all_evaluations = []
    results = {}
    
    for name, choices in policies.items():
        evaluation = evaluate_policy(choices, targets, name)
        all_evaluations.append(evaluation)
        results[name] = compute_metrics(evaluation)
    
    # Combine evaluations
    combined = pd.concat(all_evaluations, ignore_index=True)
    
    # Save output
    output_path = project_root / "storage" / "research" / "datasets" / "evaluation_transfer_in.csv"
    combined.to_csv(output_path, index=False)
    print(f"\n4. Saved: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Transfer-IN Policy Comparison")
    print("=" * 70)
    print(f"\n{'Policy':<35} {'Mean':>10} {'Median':>10} {'% ≥10':>10}")
    print("-" * 70)
    
    for name in ["stage_7a_p_play_x_mu_points", "baseline_a_mu_points", 
                 "baseline_b_random", "baseline_c_points_per_90"]:
        m = results[name]
        print(f"{name:<35} {m['mean_regret']:>10.2f} {m['median_regret']:>10.1f} {m['pct_high_regret']:>10.1%}")
    
    print("-" * 70)
    
    # Hypothesis test
    policy_regret = results["stage_7a_p_play_x_mu_points"]["mean_regret"]
    baseline_regret = results["baseline_a_mu_points"]["mean_regret"]
    delta = policy_regret - baseline_regret
    
    print(f"\nHypothesis test: p_play × mu_points vs mu_points alone")
    print(f"  Policy regret:   {policy_regret:.2f}")
    print(f"  Baseline regret: {baseline_regret:.2f}")
    print(f"  Delta:           {delta:+.2f}")
    
    if delta < 0:
        print(f"\n✅ Availability-adjusted EV reduces regret by {abs(delta):.2f} pts/GW")
    elif delta > 0:
        print(f"\n❌ Availability-adjusted EV increases regret by {delta:.2f} pts/GW")
    else:
        print(f"\n➖ No difference between policies")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
