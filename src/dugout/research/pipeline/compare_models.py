"""
Three-Way Model Comparison: Baseline vs Production vs Fixture Context v2

Compares decision metrics across:
1. Research Baseline (Stage 5 beliefs)
2. Production Model (current two_stage model)  
3. Fixture Context v2 (new opponent context features)

Evaluation on:
- Captain decision regret
- Transfer-in decision regret
- Free Hit capture rate (if applicable)
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from dugout.production.data.reader import DataReader
from dugout.production.features.builder import FeatureBuilder
from dugout.production.models.predict import predict_points


def load_research_beliefs(project_root: Path, variant: str) -> pd.DataFrame:
    """Load research beliefs CSV and compute expected_points if needed."""
    if variant == "baseline":
        path = project_root / "storage" / "research" / "datasets" / "beliefs.csv"
    elif variant == "fixture_v2":
        path = project_root / "storage" / "research" / "datasets" / "beliefs_fixture_v2_augmented.csv"
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    df = pd.read_csv(path)
    
    # Compute expected_points if not present
    if "expected_points" not in df.columns:
        df["expected_points"] = df["p_play"] * df["mu_points"]
    
    return df


def generate_production_predictions(project_root: Path, model_variant: str = "base") -> pd.DataFrame:
    """
    Generate predictions using current production model.
    
    Returns DataFrame with player_id, gw, expected_points
    """
    reader = DataReader()
    raw_df = reader.get_all_gw_data()
    
    results = []
    available_gws = sorted(raw_df["gw"].unique())
    
    for target_gw in available_gws:
        if target_gw < 6:  # Need history for features
            continue
            
        history_gw = target_gw - 1
        history_df = raw_df[raw_df["gw"] <= history_gw].copy()
        
        # Get fixtures for target GW
        fixtures = reader.get_fixtures(gw=target_gw)
        fixture_map = {}
        for f in fixtures:
            fixture_map[f["team_h"]] = True
            fixture_map[f["team_a"]] = False
        
        # Build features
        fb = FeatureBuilder()
        latest_df = fb.build_for_prediction(history_df, fixture_map)
        
        # Predict
        latest_df["expected_points"] = predict_points(latest_df, model_variant=model_variant)
        
        results.append(latest_df[["player_id", "expected_points"]].assign(gw=target_gw))
    
    return pd.concat(results, ignore_index=True)


def evaluate_captain_regret(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    gw_range: tuple[int, int],
    points_col: str = "expected_points",
) -> dict:
    """
    Evaluate captain decision regret.
    
    Decision rule: argmax(expected_points) per GW
    Regret: optimal_points - chosen_points (doubled)
    """
    merged = predictions.merge(
        actuals[["player_id", "gw", "actual_points"]],
        on=["player_id", "gw"],
        how="inner",
    )
    
    results = []
    
    for gw in range(gw_range[0], gw_range[1] + 1):
        gw_df = merged[merged["gw"] == gw]
        if len(gw_df) == 0:
            continue
        
        # Decision: argmax(expected_points)
        chosen_idx = gw_df[points_col].idxmax()
        chosen_points = gw_df.loc[chosen_idx, "actual_points"] * 2
        
        # Oracle: argmax(actual_points)
        optimal_idx = gw_df["actual_points"].idxmax()
        optimal_points = gw_df.loc[optimal_idx, "actual_points"] * 2
        
        regret = optimal_points - chosen_points
        results.append({
            "gw": gw,
            "chosen_points": chosen_points,
            "optimal_points": optimal_points,
            "regret": regret,
        })
    
    if not results:
        return {"mean_regret": float("nan"), "n_gws": 0}
    
    df_results = pd.DataFrame(results)
    return {
        "mean_regret": df_results["regret"].mean(),
        "total_regret": df_results["regret"].sum(),
        "high_regret_rate": (df_results["regret"] >= 10).mean(),
        "hit_rate": (df_results["regret"] == 0).mean(),
        "n_gws": len(df_results),
    }


def evaluate_transfer_regret(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    gw_range: tuple[int, int],
    points_col: str = "expected_points",
) -> dict:
    """
    Evaluate transfer-in decision regret.
    
    Decision rule: argmax(expected_points) per GW
    Regret: oracle_points - chosen_points
    """
    merged = predictions.merge(
        actuals[["player_id", "gw", "actual_points"]],
        on=["player_id", "gw"],
        how="inner",
    )
    
    results = []
    
    for gw in range(gw_range[0], gw_range[1] + 1):
        gw_df = merged[merged["gw"] == gw]
        if len(gw_df) == 0:
            continue
        
        # Decision: argmax(expected_points)
        chosen_idx = gw_df[points_col].idxmax()
        chosen_points = gw_df.loc[chosen_idx, "actual_points"]
        
        # Oracle: argmax(actual_points)
        optimal_idx = gw_df["actual_points"].idxmax()
        optimal_points = gw_df.loc[optimal_idx, "actual_points"]
        
        regret = optimal_points - chosen_points
        results.append({
            "gw": gw,
            "chosen_points": chosen_points,
            "optimal_points": optimal_points,
            "regret": regret,
        })
    
    if not results:
        return {"mean_regret": float("nan"), "n_gws": 0}
    
    df_results = pd.DataFrame(results)
    return {
        "mean_regret": df_results["regret"].mean(),
        "total_regret": df_results["regret"].sum(),
        "high_regret_rate": (df_results["regret"] >= 10).mean(),
        "hit_rate": (df_results["regret"] == 0).mean(),
        "n_gws": len(df_results),
    }


def main():
    project_root = Path(__file__).resolve().parents[4]
    
    print("=" * 70)
    print("THREE-WAY MODEL COMPARISON")
    print("=" * 70)
    
    # Load actuals (targets)
    print("\n[1/4] Loading actuals...")
    targets = pd.read_csv(
        project_root / "storage" / "research" / "datasets" / "targets.csv"
    )
    actuals = targets[["player_id", "gw", "y_points"]].rename(
        columns={"y_points": "actual_points"}
    )
    print(f"  Loaded {len(actuals):,} rows")
    
    # Load research beliefs
    print("\n[2/4] Loading research beliefs...")
    beliefs_baseline = load_research_beliefs(project_root, "baseline")
    beliefs_fixture_v2 = load_research_beliefs(project_root, "fixture_v2")
    print(f"  Baseline: {len(beliefs_baseline):,} rows")
    print(f"  Fixture v2: {len(beliefs_fixture_v2):,} rows")
    
    # Generate production predictions
    print("\n[3/4] Generating production model predictions...")
    prod_predictions = generate_production_predictions(project_root, model_variant="base")
    print(f"  Production: {len(prod_predictions):,} rows")
    
    # Evaluate all three
    print("\n[4/4] Evaluating decision metrics...")
    gw_range = (6, 22)  # Common GW range
    
    # Prepare DataFrames with consistent column names
    baseline_df = beliefs_baseline[["player_id", "gw", "expected_points"]].copy()
    fixture_v2_df = beliefs_fixture_v2[["player_id", "gw", "expected_points"]].copy()
    production_df = prod_predictions.copy()
    
    results = {}
    
    # Captain evaluation
    print("\n" + "-" * 70)
    print("CAPTAIN DECISION (argmax expected_points, regret = oracle - chosen × 2)")
    print("-" * 70)
    
    captain_baseline = evaluate_captain_regret(baseline_df, actuals, gw_range)
    captain_production = evaluate_captain_regret(production_df, actuals, gw_range)
    captain_fixture_v2 = evaluate_captain_regret(fixture_v2_df, actuals, gw_range)
    
    results["captain"] = {
        "baseline": captain_baseline,
        "production": captain_production,
        "fixture_v2": captain_fixture_v2,
    }
    
    print(f"\n{'Model':<20} {'Mean Regret':>12} {'High Regret%':>12} {'Hit Rate':>10}")
    print("-" * 56)
    print(f"{'Research Baseline':<20} {captain_baseline['mean_regret']:>12.2f} {captain_baseline['high_regret_rate']:>11.1%} {captain_baseline['hit_rate']:>9.1%}")
    print(f"{'Production (base)':<20} {captain_production['mean_regret']:>12.2f} {captain_production['high_regret_rate']:>11.1%} {captain_production['hit_rate']:>9.1%}")
    print(f"{'Fixture Context v2':<20} {captain_fixture_v2['mean_regret']:>12.2f} {captain_fixture_v2['high_regret_rate']:>11.1%} {captain_fixture_v2['hit_rate']:>9.1%}")
    
    # Transfer evaluation
    print("\n" + "-" * 70)
    print("TRANSFER-IN DECISION (argmax expected_points, regret = oracle - chosen)")
    print("-" * 70)
    
    transfer_baseline = evaluate_transfer_regret(baseline_df, actuals, gw_range)
    transfer_production = evaluate_transfer_regret(production_df, actuals, gw_range)
    transfer_fixture_v2 = evaluate_transfer_regret(fixture_v2_df, actuals, gw_range)
    
    results["transfer"] = {
        "baseline": transfer_baseline,
        "production": transfer_production,
        "fixture_v2": transfer_fixture_v2,
    }
    
    print(f"\n{'Model':<20} {'Mean Regret':>12} {'High Regret%':>12} {'Hit Rate':>10}")
    print("-" * 56)
    print(f"{'Research Baseline':<20} {transfer_baseline['mean_regret']:>12.2f} {transfer_baseline['high_regret_rate']:>11.1%} {transfer_baseline['hit_rate']:>9.1%}")
    print(f"{'Production (base)':<20} {transfer_production['mean_regret']:>12.2f} {transfer_production['high_regret_rate']:>11.1%} {transfer_production['hit_rate']:>9.1%}")
    print(f"{'Fixture Context v2':<20} {transfer_fixture_v2['mean_regret']:>12.2f} {transfer_fixture_v2['high_regret_rate']:>11.1%} {transfer_fixture_v2['hit_rate']:>9.1%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    best_captain = min(
        [("Research Baseline", captain_baseline["mean_regret"]),
         ("Production (base)", captain_production["mean_regret"]),
         ("Fixture Context v2", captain_fixture_v2["mean_regret"])],
        key=lambda x: x[1]
    )
    
    best_transfer = min(
        [("Research Baseline", transfer_baseline["mean_regret"]),
         ("Production (base)", transfer_production["mean_regret"]),
         ("Fixture Context v2", transfer_fixture_v2["mean_regret"])],
        key=lambda x: x[1]
    )
    
    print(f"\n  Best Captain Model:  {best_captain[0]} ({best_captain[1]:.2f} pts/GW regret)")
    print(f"  Best Transfer Model: {best_transfer[0]} ({best_transfer[1]:.2f} pts/GW regret)")
    
    # Improvement vs production
    print("\n  Fixture v2 vs Production:")
    captain_delta = captain_production["mean_regret"] - captain_fixture_v2["mean_regret"]
    transfer_delta = transfer_production["mean_regret"] - transfer_fixture_v2["mean_regret"]
    
    print(f"    Captain:  {captain_delta:+.2f} pts/GW {'✓' if captain_delta > 0 else '✗'}")
    print(f"    Transfer: {transfer_delta:+.2f} pts/GW {'✓' if transfer_delta > 0 else '✗'}")
    
    print("=" * 70)
    
    # Save results
    reports_dir = project_root / "storage" / "research" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    with open(reports_dir / "three_way_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to {reports_dir / 'three_way_comparison.json'}")


if __name__ == "__main__":
    main()
