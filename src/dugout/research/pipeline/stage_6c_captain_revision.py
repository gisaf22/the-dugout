"""
Stage 6c — Captain Policy Revision

Updates the captain policy based on empirical evidence from Stage 6b.
Uses mu_points alone (no availability weighting) for captain selection.

Evidence-based correction:
    Original: score = p_play × mu_points  (6.91 mean regret)
    Revised:  score = mu_points           (6.23 mean regret)

This stage:
    1. Applies the revised policy
    2. Evaluates using same regret framework
    3. Compares against original policy
    4. Preserves prior artifacts

Inputs:
    storage/datasets/beliefs.csv
    storage/datasets/targets.csv
    storage/datasets/evaluation_captain.csv (original, for comparison)

Outputs:
    storage/datasets/evaluation_captain_revised.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Revised Policy
# -----------------------------------------------------------------------------


def compute_revised_captain_score(beliefs: pd.DataFrame) -> pd.DataFrame:
    """
    Revised captain score: mu_points only.
    
    Based on Stage 6b evidence that availability weighting
    increases regret for captain selection.
    """
    df = beliefs[["player_id", "gw"]].copy()
    df["score"] = beliefs["mu_points"].fillna(0)
    return df


def build_captain_decisions(beliefs: pd.DataFrame) -> pd.DataFrame:
    """
    Select captain for each GW using revised policy.
    
    score = mu_points
    captain = argmax(score) per GW
    """
    scores = compute_revised_captain_score(beliefs)
    
    # Find max score per GW
    idx_max = scores.groupby("gw")["score"].idxmax()
    
    # Mark chosen
    scores["chosen"] = False
    scores.loc[idx_max, "chosen"] = True
    
    return scores[["gw", "player_id", "score", "chosen"]]


# -----------------------------------------------------------------------------
# Evaluation (same as Stage 6a)
# -----------------------------------------------------------------------------


def evaluate_captain_policy(
    decisions: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate captain decisions using regret.
    Identical to Stage 6a evaluation logic.
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
    
    return evaluation[
        ["gw", "chosen_player_id", "chosen_points", "optimal_player_id", "optimal_points", "regret"]
    ]


def compute_metrics(evaluation: pd.DataFrame) -> dict:
    """Compute aggregate regret metrics."""
    regret = evaluation["regret"]
    return {
        "mean_regret": float(regret.mean()),
        "median_regret": float(regret.median()),
        "pct_ge_10": float((regret >= 10).mean() * 100),
        "total_regret": int(regret.sum()),
    }


# -----------------------------------------------------------------------------
# Pipeline Entry Point
# -----------------------------------------------------------------------------


def run_captain_revision(project_root: Path) -> None:
    """
    Execute Stage 6c: Captain Policy Revision.
    """
    print("=" * 50)
    print("Stage 6c — Captain Policy Revision")
    print("=" * 50)
    
    # Load inputs
    print("\n1. Loading frozen artifacts...")
    beliefs_path = project_root / "storage" / "research" / "datasets" / "beliefs.csv"
    targets_path = project_root / "storage" / "research" / "datasets" / "targets.csv"
    eval_original_path = project_root / "storage" / "research" / "datasets" / "evaluation_captain.csv"
    
    beliefs = pd.read_csv(beliefs_path)
    targets = pd.read_csv(targets_path)
    eval_original = pd.read_csv(eval_original_path)
    
    print(f"   Beliefs: {len(beliefs):,} rows")
    print(f"   Targets: {len(targets):,} rows")
    print(f"   Original evaluation: {len(eval_original)} GWs")
    
    # Build revised decisions
    print("\n2. Applying revised policy...")
    print("   Revised score: mu_points (no availability weighting)")
    
    decisions = build_captain_decisions(beliefs)
    n_captains = decisions["chosen"].sum()
    print(f"   Selected {n_captains} captains")
    
    # Evaluate revised policy
    print("\n3. Evaluating revised policy...")
    eval_revised = evaluate_captain_policy(decisions, targets)
    
    # Compute metrics
    metrics_original = compute_metrics(eval_original)
    metrics_revised = compute_metrics(eval_revised)
    
    # Save output
    print("\n4. Saving outputs...")
    output_path = project_root / "storage" / "research" / "datasets" / "evaluation_captain_revised.csv"
    eval_revised.to_csv(output_path, index=False)
    print(f"   Saved {output_path}")
    
    # Comparison
    print("\n" + "=" * 50)
    print("Stage 6c Complete — Policy Comparison")
    print("=" * 50)
    
    print("\n{:<25} {:>15} {:>15}".format("Metric", "Original", "Revised"))
    print("-" * 60)
    
    print("{:<25} {:>15.2f} {:>15.2f}".format(
        "Mean regret",
        metrics_original["mean_regret"],
        metrics_revised["mean_regret"]
    ))
    print("{:<25} {:>15.1f} {:>15.1f}".format(
        "Median regret",
        metrics_original["median_regret"],
        metrics_revised["median_regret"]
    ))
    print("{:<25} {:>14.1f}% {:>14.1f}%".format(
        "% GW with regret ≥ 10",
        metrics_original["pct_ge_10"],
        metrics_revised["pct_ge_10"]
    ))
    print("{:<25} {:>15} {:>15}".format(
        "Total regret",
        metrics_original["total_regret"],
        metrics_revised["total_regret"]
    ))
    
    # Summary
    print("\n" + "-" * 60)
    
    improvement = metrics_original["mean_regret"] - metrics_revised["mean_regret"]
    
    if improvement > 0:
        print(f"\n✅ Revised policy reduces mean regret by {improvement:.2f} pts/GW")
        print("\nPolicy correction validated:")
        print("  Original: score = p_play × mu_points")
        print("  Revised:  score = mu_points")
    else:
        print(f"\n❌ Revised policy increases regret by {-improvement:.2f} pts/GW")
        print("   Policy correction NOT validated.")
    
    # Artifact preservation note
    print("\n" + "-" * 60)
    print("Artifacts preserved:")
    print("  evaluation_captain.csv          (Stage 6a original)")
    print("  evaluation_captain_baselines.csv (Stage 6b comparison)")
    print("  evaluation_captain_revised.csv  (Stage 6c revised)")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for Stage 6c."""
    project_root = Path(__file__).resolve().parents[4]
    run_captain_revision(project_root)


if __name__ == "__main__":
    main()
