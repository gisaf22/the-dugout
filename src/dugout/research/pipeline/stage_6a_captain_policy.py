"""
Stage 6a — Captain Selection Under Uncertainty

Converts frozen belief estimates into a captain selection policy and evaluates
decision quality using regret-based metrics.

This stage consumes beliefs and produces decisions + evaluation.
It does NOT train models, engineer features, or tune parameters.

Decision Policy (v1):
    score = p_play * mu_points
    captain = argmax(score) per gameweek

Evaluation Metric:
    regret = max(actual_points) - actual_points(chosen_player)
    where optimal is computed among players who actually played (y_play==1)

Inputs:
    storage/datasets/beliefs.csv
    storage/datasets/targets.csv (for evaluation only)

Outputs:
    storage/datasets/decisions_captain.csv
    storage/datasets/evaluation_captain.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Decision Policy
# -----------------------------------------------------------------------------


def compute_captain_score(beliefs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute captain selection score for each (player_id, gw).

    Score = p_play * mu_points

    Args:
        beliefs: DataFrame with columns [player_id, gw, p_play, mu_points, ...]

    Returns:
        DataFrame with columns [player_id, gw, score]
    """
    df = beliefs[["player_id", "gw"]].copy()

    # Score = p_play * mu_points
    # Handle missing mu_points (players with no prior appearances)
    p_play = beliefs["p_play"].fillna(0)
    mu_points = beliefs["mu_points"].fillna(0)

    df["score"] = p_play * mu_points

    return df


def build_captain_decisions(beliefs: pd.DataFrame) -> pd.DataFrame:
    """
    Select a captain for each gameweek using belief estimates.

    For each GW:
        1. Compute score = p_play * mu_points for all players
        2. Select player with maximum score
        3. Mark as chosen=True

    Args:
        beliefs: DataFrame from beliefs.csv

    Returns:
        DataFrame with columns [gw, player_id, score, chosen]
        Exactly one chosen=True per gw.
    """
    # Compute scores
    scores = compute_captain_score(beliefs)

    # Find max score per GW
    idx_max = scores.groupby("gw")["score"].idxmax()

    # Mark chosen
    scores["chosen"] = False
    scores.loc[idx_max, "chosen"] = True

    # Sort for readability
    decisions = scores.sort_values(["gw", "chosen", "score"], ascending=[True, False, False])

    return decisions[["gw", "player_id", "score", "chosen"]]


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def evaluate_captain_policy(
    decisions: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate captain decisions using regret-based metrics.

    For each gameweek:
        regret = max(actual_points among played) - actual_points(chosen)

    Args:
        decisions: DataFrame with [gw, player_id, score, chosen]
        targets: DataFrame with [player_id, gw, y_play, y_points]

    Returns:
        DataFrame with columns:
            gw, chosen_player_id, chosen_points, optimal_player_id,
            optimal_points, regret
    """
    # Get chosen captain per GW
    chosen = decisions[decisions["chosen"] == True][["gw", "player_id"]].copy()
    chosen = chosen.rename(columns={"player_id": "chosen_player_id"})

    # Merge with targets to get actual points
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

    # Handle cases where chosen player didn't play (0 points)
    evaluation["chosen_points"] = evaluation["chosen_points"].fillna(0).astype(int)
    evaluation["optimal_points"] = evaluation["optimal_points"].fillna(0).astype(int)

    # Compute regret
    evaluation["regret"] = evaluation["optimal_points"] - evaluation["chosen_points"]

    return evaluation[
        ["gw", "chosen_player_id", "chosen_points", "optimal_player_id", "optimal_points", "regret"]
    ]


def compute_aggregate_metrics(evaluation: pd.DataFrame) -> dict:
    """
    Compute aggregate regret metrics.

    Returns:
        Dict with mean_regret, median_regret, total_regret, n_gameweeks
    """
    regret = evaluation["regret"]
    return {
        "mean_regret": float(regret.mean()),
        "median_regret": float(regret.median()),
        "total_regret": int(regret.sum()),
        "n_gameweeks": len(regret),
        "min_regret": int(regret.min()),
        "max_regret": int(regret.max()),
        "zero_regret_rate": float((regret == 0).mean()),
    }


# -----------------------------------------------------------------------------
# Pipeline Entry Point
# -----------------------------------------------------------------------------


def run_captain_policy(project_root: Path) -> None:
    """
    Execute Stage 6a: Captain Selection Policy.

    Reads beliefs, produces captain decisions, and evaluates policy.
    """
    print("=" * 50)
    print("Stage 6a — Captain Selection Under Uncertainty")
    print("=" * 50)

    # Load inputs
    print("\n1. Loading frozen artifacts...")
    beliefs_path = project_root / "storage" / "research" / "datasets" / "beliefs.csv"
    targets_path = project_root / "storage" / "research" / "datasets" / "targets.csv"

    beliefs = pd.read_csv(beliefs_path)
    targets = pd.read_csv(targets_path)

    print(f"   Beliefs: {len(beliefs):,} rows, GW {beliefs['gw'].min()}-{beliefs['gw'].max()}")
    print(f"   Targets: {len(targets):,} rows")

    # Validate required columns
    required_belief_cols = ["player_id", "gw", "p_play", "mu_points"]
    missing = [c for c in required_belief_cols if c not in beliefs.columns]
    if missing:
        raise ValueError(f"Missing columns in beliefs.csv: {missing}")

    required_target_cols = ["player_id", "gw", "y_play", "y_points"]
    missing = [c for c in required_target_cols if c not in targets.columns]
    if missing:
        raise ValueError(f"Missing columns in targets.csv: {missing}")

    # Build decisions
    print("\n2. Building captain decisions...")
    decisions = build_captain_decisions(beliefs)

    n_gws = decisions["gw"].nunique()
    n_chosen = decisions["chosen"].sum()
    print(f"   {n_gws} gameweeks, {n_chosen} captains selected")

    # Sanity check: one captain per GW
    captains_per_gw = decisions.groupby("gw")["chosen"].sum()
    if (captains_per_gw != 1).any():
        bad_gws = captains_per_gw[captains_per_gw != 1].index.tolist()
        raise ValueError(f"Multiple/missing captains in GWs: {bad_gws}")

    # Evaluate policy
    print("\n3. Evaluating captain policy...")
    evaluation = evaluate_captain_policy(decisions, targets)

    metrics = compute_aggregate_metrics(evaluation)
    print(f"   Mean regret: {metrics['mean_regret']:.2f} points")
    print(f"   Median regret: {metrics['median_regret']:.1f} points")
    print(f"   Zero regret rate: {metrics['zero_regret_rate']:.1%}")

    # Save outputs
    print("\n4. Saving outputs...")
    datasets_dir = project_root / "storage" / "research" / "datasets"

    decisions_path = datasets_dir / "decisions_captain.csv"
    decisions.to_csv(decisions_path, index=False)
    print(f"   Saved {decisions_path}")

    evaluation_path = datasets_dir / "evaluation_captain.csv"
    evaluation.to_csv(evaluation_path, index=False)
    print(f"   Saved {evaluation_path}")

    # Summary
    print("\n" + "=" * 50)
    print("Stage 6a Complete")
    print("=" * 50)

    print("\nDecision Policy: score = p_play × mu_points")
    print(f"\nCaptain selections: {n_chosen} across GWs {beliefs['gw'].min()}-{beliefs['gw'].max()}")

    print("\nRegret Analysis:")
    print(f"  Mean:   {metrics['mean_regret']:.2f} pts")
    print(f"  Median: {metrics['median_regret']:.1f} pts")
    print(f"  Total:  {metrics['total_regret']} pts over {metrics['n_gameweeks']} GWs")
    print(f"  Range:  {metrics['min_regret']}-{metrics['max_regret']} pts")
    print(f"  Perfect picks: {metrics['zero_regret_rate']:.1%}")

    # Show top regret GWs
    print("\nHighest regret gameweeks:")
    top_regret = evaluation.nlargest(5, "regret")[
        ["gw", "chosen_points", "optimal_points", "regret"]
    ]
    print(top_regret.to_string(index=False))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for Stage 6a."""
    project_root = Path(__file__).resolve().parents[4]
    run_captain_policy(project_root)


if __name__ == "__main__":
    main()
