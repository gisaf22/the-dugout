"""
Stage 6b — Captain Baseline Comparison

Compares the belief-driven captain policy (Stage 6a) against simple baseline
captain policies using regret-based evaluation.

This stage is comparative only. No learning, tuning, or feature engineering.

Baseline Policies:
    A. Mean Performance Only:  score = mu_points
    B. Availability Only:      score = p_play
    C. Empirical Scorer:       score = points_per_90_5 (if available)

Evaluation:
    regret = max(actual_points) - actual_points(chosen_player)
    Same logic as Stage 6a.

Inputs:
    storage/datasets/beliefs.csv
    storage/datasets/targets.csv
    storage/datasets/evaluation_captain.csv (Stage 6a output)
    storage/datasets/features_performance.csv (for Baseline C)

Outputs:
    storage/datasets/evaluation_captain_baselines.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Baseline Policy Definitions
# -----------------------------------------------------------------------------


def compute_baseline_a_score(beliefs: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline A — Mean Performance Only.
    
    Score = mu_points
    
    Ignores availability. Tests value of modeling participation.
    """
    df = beliefs[["player_id", "gw"]].copy()
    df["score"] = beliefs["mu_points"].fillna(0)
    return df


def compute_baseline_b_score(beliefs: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline B — Availability Only.
    
    Score = p_play
    
    Always captain the most likely starter.
    """
    df = beliefs[["player_id", "gw"]].copy()
    df["score"] = beliefs["p_play"].fillna(0)
    return df


def compute_baseline_c_score(
    beliefs: pd.DataFrame, 
    performance_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Baseline C — Empirical Scorer.
    
    Score = points_per_90_5
    
    Uses historical points per 90 minutes from last 5 games.
    """
    df = beliefs[["player_id", "gw"]].copy()
    
    # Merge with performance features
    df = df.merge(
        performance_features[["player_id", "gw", "points_per_90_5"]],
        on=["player_id", "gw"],
        how="left"
    )
    df["score"] = df["points_per_90_5"].fillna(0)
    df = df.drop(columns=["points_per_90_5"])
    
    return df


# -----------------------------------------------------------------------------
# Decision Selection
# -----------------------------------------------------------------------------


def select_captain(scores: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    """
    Select captain for each GW based on scores.
    
    Returns DataFrame with [policy_name, gw, player_id, score, chosen]
    """
    # Find max score per GW
    idx_max = scores.groupby("gw")["score"].idxmax()
    
    # Mark chosen
    scores["chosen"] = False
    scores.loc[idx_max, "chosen"] = True
    
    # Add policy name
    scores["policy_name"] = policy_name
    
    return scores[["policy_name", "gw", "player_id", "score", "chosen"]]


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def evaluate_policy(
    decisions: pd.DataFrame,
    targets: pd.DataFrame,
    policy_name: str,
) -> pd.DataFrame:
    """
    Evaluate a captain policy using regret.
    
    Same logic as Stage 6a.
    """
    # Get chosen captain per GW
    chosen = decisions[decisions["chosen"] == True][["gw", "player_id"]].copy()
    chosen = chosen.rename(columns={"player_id": "chosen_player_id"})
    
    # Get targets for players who played
    targets_played = targets[targets["y_play"] == 1].copy()
    
    # Get chosen player's actual points
    chosen = chosen.merge(
        targets[["player_id", "gw", "y_points"]].rename(
            columns={"player_id": "chosen_player_id", "y_points": "chosen_points"}
        ),
        on=["gw", "chosen_player_id"],
        how="left",
    )
    
    # Find optimal (max points) per GW among players who played
    optimal = (
        targets_played.loc[targets_played.groupby("gw")["y_points"].idxmax()]
        [["gw", "player_id", "y_points"]]
        .rename(columns={"player_id": "optimal_player_id", "y_points": "optimal_points"})
    )
    
    # Merge chosen with optimal
    evaluation = chosen.merge(optimal, on="gw", how="left")
    
    # Handle missing
    evaluation["chosen_points"] = evaluation["chosen_points"].fillna(0).astype(int)
    evaluation["optimal_points"] = evaluation["optimal_points"].fillna(0).astype(int)
    
    # Compute regret
    evaluation["regret"] = evaluation["optimal_points"] - evaluation["chosen_points"]
    
    # Add policy name
    evaluation["policy_name"] = policy_name
    
    return evaluation[
        ["policy_name", "gw", "chosen_player_id", "chosen_points", 
         "optimal_player_id", "optimal_points", "regret"]
    ]


def compute_aggregate_metrics(evaluation: pd.DataFrame, policy_name: str) -> dict:
    """Compute aggregate regret metrics for a policy."""
    regret = evaluation["regret"]
    return {
        "policy_name": policy_name,
        "mean_regret": float(regret.mean()),
        "median_regret": float(regret.median()),
        "pct_regret_ge_10": float((regret >= 10).mean() * 100),
        "total_regret": int(regret.sum()),
    }


# -----------------------------------------------------------------------------
# Pipeline Entry Point
# -----------------------------------------------------------------------------


def run_captain_baselines(project_root: Path) -> None:
    """
    Execute Stage 6b: Captain Baseline Comparison.
    """
    print("=" * 50)
    print("Stage 6b — Captain Baseline Comparison")
    print("=" * 50)
    
    # Load inputs
    print("\n1. Loading frozen artifacts...")
    beliefs_path = project_root / "storage" / "research" / "datasets" / "beliefs.csv"
    targets_path = project_root / "storage" / "research" / "datasets" / "targets.csv"
    eval_6a_path = project_root / "storage" / "research" / "datasets" / "evaluation_captain.csv"
    perf_features_path = project_root / "storage" / "research" / "datasets" / "features_performance.csv"
    
    beliefs = pd.read_csv(beliefs_path)
    targets = pd.read_csv(targets_path)
    eval_6a = pd.read_csv(eval_6a_path)
    
    print(f"   Beliefs: {len(beliefs):,} rows")
    print(f"   Targets: {len(targets):,} rows")
    print(f"   Stage 6a evaluation: {len(eval_6a)} GWs")
    
    # Check for performance features (Baseline C)
    has_perf_features = perf_features_path.exists()
    if has_perf_features:
        perf_features = pd.read_csv(perf_features_path)
        print(f"   Performance features: {len(perf_features):,} rows")
    else:
        print("   Performance features: not found (skipping Baseline C)")
    
    # Define baselines
    print("\n2. Computing baseline policies...")
    
    all_evaluations = []
    all_metrics = []
    
    # Baseline A — Mean Performance Only
    print("   Baseline A: score = mu_points")
    scores_a = compute_baseline_a_score(beliefs)
    decisions_a = select_captain(scores_a, "baseline_a_mu_points")
    eval_a = evaluate_policy(decisions_a, targets, "baseline_a_mu_points")
    metrics_a = compute_aggregate_metrics(eval_a, "baseline_a_mu_points")
    all_evaluations.append(eval_a)
    all_metrics.append(metrics_a)
    
    # Baseline B — Availability Only
    print("   Baseline B: score = p_play")
    scores_b = compute_baseline_b_score(beliefs)
    decisions_b = select_captain(scores_b, "baseline_b_p_play")
    eval_b = evaluate_policy(decisions_b, targets, "baseline_b_p_play")
    metrics_b = compute_aggregate_metrics(eval_b, "baseline_b_p_play")
    all_evaluations.append(eval_b)
    all_metrics.append(metrics_b)
    
    # Baseline C — Empirical Scorer (if available)
    if has_perf_features and "points_per_90_5" in perf_features.columns:
        print("   Baseline C: score = points_per_90_5")
        scores_c = compute_baseline_c_score(beliefs, perf_features)
        decisions_c = select_captain(scores_c, "baseline_c_points_per_90")
        eval_c = evaluate_policy(decisions_c, targets, "baseline_c_points_per_90")
        metrics_c = compute_aggregate_metrics(eval_c, "baseline_c_points_per_90")
        all_evaluations.append(eval_c)
        all_metrics.append(metrics_c)
    
    # Add Stage 6a results for comparison
    eval_6a["policy_name"] = "stage_6a_belief_policy"
    all_evaluations.append(eval_6a)
    metrics_6a = compute_aggregate_metrics(eval_6a, "stage_6a_belief_policy")
    all_metrics.append(metrics_6a)
    
    # Combine all evaluations
    combined_eval = pd.concat(all_evaluations, ignore_index=True)
    
    # Save outputs
    print("\n3. Saving outputs...")
    output_path = project_root / "storage" / "research" / "datasets" / "evaluation_captain_baselines.csv"
    combined_eval.to_csv(output_path, index=False)
    print(f"   Saved {output_path}")
    
    # Print aggregate metrics
    print("\n" + "=" * 50)
    print("Stage 6b Complete — Policy Comparison")
    print("=" * 50)
    
    print("\n{:<30} {:>12} {:>12} {:>12}".format(
        "Policy", "Mean Regret", "Median", "% GW ≥ 10"
    ))
    print("-" * 70)
    
    # Sort by mean regret (best first)
    sorted_metrics = sorted(all_metrics, key=lambda x: x["mean_regret"])
    
    for m in sorted_metrics:
        print("{:<30} {:>12.2f} {:>12.1f} {:>12.1f}%".format(
            m["policy_name"],
            m["mean_regret"],
            m["median_regret"],
            m["pct_regret_ge_10"]
        ))
    
    # Summary statement
    print("\n" + "-" * 70)
    belief_policy = next(m for m in sorted_metrics if m["policy_name"] == "stage_6a_belief_policy")
    best_baseline = next(m for m in sorted_metrics if m["policy_name"] != "stage_6a_belief_policy")
    
    if belief_policy["mean_regret"] < best_baseline["mean_regret"]:
        improvement = best_baseline["mean_regret"] - belief_policy["mean_regret"]
        print(f"\n✅ Belief-driven policy reduces mean regret by {improvement:.2f} pts vs best baseline")
    else:
        gap = belief_policy["mean_regret"] - best_baseline["mean_regret"]
        print(f"\n❌ Belief-driven policy has {gap:.2f} pts MORE regret than best baseline")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for Stage 6b."""
    project_root = Path(__file__).resolve().parents[4]
    run_captain_baselines(project_root)


if __name__ == "__main__":
    main()
