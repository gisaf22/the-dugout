"""
Stage 5c — Belief Models with Fixture Context v2 (Ablation)

Controlled ablation experiment to test whether adding opponent context
features (xG/xGC-based) improves belief quality and downstream decision performance.

Key Design:
    - Toggle: use_fixture_context_v2=True/False for clean ablation
    - Role-aware features: attacking context for MID/FWD, defensive for DEF/GKP
    - Participation models: UNCHANGED (no fixture features)
    - Performance models: ADD fixture context v2 features when toggled

Inputs:
    storage/research/datasets/targets.csv
    storage/research/datasets/features_participation.csv
    storage/research/datasets/features_performance.csv
    storage/research/datasets/features_fixture_context_v2.csv

Outputs:
    storage/research/datasets/beliefs_fixture_v2_baseline.csv
    storage/research/datasets/beliefs_fixture_v2_augmented.csv

Evaluation:
    - Belief-level metrics (log-loss, MAE)
    - Decision-level regret (captain, transfer, free hit)
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

PARTICIPATION_FEATURES = [
    "p_play_hat",
    "p60_hat",
    "mins_std_5",
    "mins_below_60_rate_5",
]

# Baseline performance features
PERFORMANCE_FEATURES_BASELINE = [
    "points_per_90_5",
    "xGI_per_90_5",
    "bonus_per_90_5",
    "ict_per_90_5",
]

# Fixture context v2 features (role-aware)
FIXTURE_CONTEXT_V2_FEATURES = [
    "opp_xgc_per90_5",  # Attacking context (MID/FWD only)
    "opp_cs_rate_5",    # Attacking context (MID/FWD only)
    "opp_xg_per90_5",   # Defensive context (DEF/GKP only)
    "match_total_xg",   # Game environment (all positions)
    "is_home",          # Home/away (all positions)
]

# Walk-forward fold boundaries
WALK_FORWARD_FOLDS = [
    {"train_end": 5, "val_start": 6, "val_end": 8}, 
    {"train_end": 8, "val_start": 9, "val_end": 11},   
    {"train_end": 11, "val_start": 12, "val_end": 14},  
    {"train_end": 14, "val_start": 15, "val_end": 17}, 
    {"train_end": 17, "val_start": 18, "val_end": 20},  
    {"train_end": 20, "val_start": 21, "val_end": 22},  
]

FINAL_TRAIN_END = 22

# LightGBM hyperparameters
LGBM_CLASSIFIER_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42,
    "verbose": -1,
}

LGBM_REGRESSOR_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42,
    "verbose": -1,
}


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def load_data(project_root: Path) -> pd.DataFrame:
    """
    Load and join targets with all feature tables including fixture context v2.
    """
    targets_path = project_root / "storage" / "research" / "datasets" / "targets.csv"
    participation_path = project_root / "storage" / "research" / "datasets" / "features_participation.csv"
    performance_path = project_root / "storage" / "research" / "datasets" / "features_performance.csv"
    fixture_v2_path = project_root / "storage" / "research" / "datasets" / "features_fixture_context_v2.csv"

    targets = pd.read_csv(targets_path)
    participation = pd.read_csv(participation_path)
    performance = pd.read_csv(performance_path)
    fixture_v2 = pd.read_csv(fixture_v2_path)

    # Join on (player_id, gw)
    df = targets.merge(participation, on=["player_id", "gw"], how="left")
    df = df.merge(performance, on=["player_id", "gw"], how="left")
    df = df.merge(fixture_v2, on=["player_id", "gw"], how="left")

    return df


# -----------------------------------------------------------------------------
# Feature Selection (Role-aware)
# -----------------------------------------------------------------------------


def get_performance_features(use_fixture_context_v2: bool) -> list[str]:
    """Get performance features based on toggle."""
    features = PERFORMANCE_FEATURES_BASELINE.copy()
    if use_fixture_context_v2:
        features.extend(FIXTURE_CONTEXT_V2_FEATURES)
    return features


# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------


def prepare_participation_data(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare data for participation models (unchanged)."""
    X = df[PARTICIPATION_FEATURES].copy()
    y = df[target_col].copy()
    X = X.fillna(0)
    return X, y


def prepare_performance_data(
    df: pd.DataFrame,
    target_col: str,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    """
    Prepare data for performance models.
    
    Filters to rows where player played (y_play==1) and handles NaN features.
    For role-aware features, NaN is expected (e.g., opp_xgc for GKP) so we
    fill with 0 (neutral value).
    """
    played_mask = df["y_play"] == 1
    df_played = df[played_mask].copy()

    X = df_played[features].copy()
    y = df_played[target_col].copy()

    # Fill NaN with 0 (neutral value for role-specific features)
    X = X.fillna(0)

    return X, y, X.index


def split_by_gw(
    df: pd.DataFrame,
    train_end: int,
    val_start: int,
    val_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by gameweek for walk-forward validation."""
    train_mask = df["gw"] <= train_end
    val_mask = (df["gw"] >= val_start) & (df["gw"] <= val_end)
    return df[train_mask], df[val_mask]


# -----------------------------------------------------------------------------
# Model Training
# -----------------------------------------------------------------------------


def train_participation_model(
    df_train: pd.DataFrame,
    target_col: str,
) -> Any:
    """Train participation model (unchanged)."""
    X_train, y_train = prepare_participation_data(df_train, target_col)
    model = LGBMClassifier(**LGBM_CLASSIFIER_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_performance_model(
    df_train: pd.DataFrame,
    target_col: str,
    features: list[str],
    is_classifier: bool = False,
) -> Any:
    """Train performance model with specified feature set."""
    X_train, y_train, _ = prepare_performance_data(df_train, target_col, features)
    
    if is_classifier:
        model = LGBMClassifier(**LGBM_CLASSIFIER_PARAMS)
    else:
        model = LGBMRegressor(**LGBM_REGRESSOR_PARAMS)
    
    model.fit(X_train, y_train)
    return model


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute log loss and Brier score."""
    y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return {
        "log_loss": log_loss(y_true, y_pred_clipped),
        "brier_score": brier_score_loss(y_true, y_pred),
        "n_samples": len(y_true),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE and bias."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "bias": (y_pred - y_true).mean(),
        "n_samples": len(y_true),
    }


# -----------------------------------------------------------------------------
# Walk-Forward Validation (Ablation)
# -----------------------------------------------------------------------------


def walk_forward_ablation(df: pd.DataFrame) -> dict:
    """
    Run walk-forward validation comparing baseline vs fixture-context-v2.
    
    Returns metrics for both variants.
    """
    results = {
        "baseline": {"mu_points": [], "p_haul": []},
        "fixture_v2": {"mu_points": [], "p_haul": []},
    }
    
    baseline_features = get_performance_features(use_fixture_context_v2=False)
    augmented_features = get_performance_features(use_fixture_context_v2=True)
    
    for fold in WALK_FORWARD_FOLDS:
        df_train, df_val = split_by_gw(
            df, fold["train_end"], fold["val_start"], fold["val_end"]
        )
        fold_label = f"GW {fold['val_start']}-{fold['val_end']}"
        
        # --- mu_points: Baseline ---
        model_base = train_performance_model(
            df_train, "y_points", baseline_features, is_classifier=False
        )
        X_val, y_val, val_idx = prepare_performance_data(
            df_val, "y_points", baseline_features
        )
        if len(X_val) > 0:
            y_pred = model_base.predict(X_val)
            metrics = compute_regression_metrics(y_val.values, y_pred)
            metrics["fold"] = fold_label
            results["baseline"]["mu_points"].append(metrics)
        
        # --- mu_points: Fixture v2 ---
        model_aug = train_performance_model(
            df_train, "y_points", augmented_features, is_classifier=False
        )
        X_val_aug, y_val_aug, _ = prepare_performance_data(
            df_val, "y_points", augmented_features
        )
        if len(X_val_aug) > 0:
            y_pred = model_aug.predict(X_val_aug)
            metrics = compute_regression_metrics(y_val_aug.values, y_pred)
            metrics["fold"] = fold_label
            results["fixture_v2"]["mu_points"].append(metrics)
        
        # --- p_haul: Baseline ---
        model_base = train_performance_model(
            df_train, "y_haul", baseline_features, is_classifier=True
        )
        X_val, y_val, _ = prepare_performance_data(
            df_val, "y_haul", baseline_features
        )
        if len(X_val) > 0:
            y_pred = model_base.predict_proba(X_val)[:, 1]
            metrics = compute_classification_metrics(y_val.values, y_pred)
            metrics["fold"] = fold_label
            results["baseline"]["p_haul"].append(metrics)
        
        # --- p_haul: Fixture v2 ---
        model_aug = train_performance_model(
            df_train, "y_haul", augmented_features, is_classifier=True
        )
        X_val_aug, y_val_aug, _ = prepare_performance_data(
            df_val, "y_haul", augmented_features
        )
        if len(X_val_aug) > 0:
            y_pred = model_aug.predict_proba(X_val_aug)[:, 1]
            metrics = compute_classification_metrics(y_val_aug.values, y_pred)
            metrics["fold"] = fold_label
            results["fixture_v2"]["p_haul"].append(metrics)
    
    return results


def summarize_ablation(results: dict) -> dict:
    """Summarize ablation results across folds."""
    summary = {}
    
    for variant in ["baseline", "fixture_v2"]:
        summary[variant] = {}
        
        # mu_points
        mu_results = results[variant]["mu_points"]
        if mu_results:
            maes = [r["mae"] for r in mu_results]
            summary[variant]["mu_points_mae"] = np.mean(maes)
            summary[variant]["mu_points_mae_std"] = np.std(maes)
        
        # p_haul
        haul_results = results[variant]["p_haul"]
        if haul_results:
            log_losses = [r["log_loss"] for r in haul_results]
            summary[variant]["p_haul_log_loss"] = np.mean(log_losses)
            summary[variant]["p_haul_log_loss_std"] = np.std(log_losses)
    
    return summary


# -----------------------------------------------------------------------------
# Final Model Training & Belief Generation
# -----------------------------------------------------------------------------


def train_final_models(
    df: pd.DataFrame,
    use_fixture_context_v2: bool = False,
) -> dict[str, Any]:
    """
    Train final models.
    
    Args:
        df: Full dataset
        use_fixture_context_v2: Toggle for fixture context v2 features
    """
    df_train = df[df["gw"] <= FINAL_TRAIN_END]
    
    features = get_performance_features(use_fixture_context_v2)
    
    models = {}
    
    # Participation models (unchanged)
    models["p_play"] = train_participation_model(df_train, "y_play")
    models["p60"] = train_participation_model(df_train, "y_60")
    
    # Performance models
    models["mu_points"] = train_performance_model(
        df_train, "y_points", features, is_classifier=False
    )
    models["p_haul"] = train_performance_model(
        df_train, "y_haul", features, is_classifier=True
    )
    
    # Store feature list for prediction
    models["_performance_features"] = features
    
    return models


def predict_beliefs(
    df: pd.DataFrame,
    models: dict[str, Any],
) -> pd.DataFrame:
    """Generate belief predictions."""
    features = models["_performance_features"]
    
    beliefs = pd.DataFrame({
        "player_id": df["player_id"],
        "gw": df["gw"],
    })
    
    # p_play
    X_part, _ = prepare_participation_data(df, "y_play")
    beliefs["p_play"] = models["p_play"].predict_proba(X_part)[:, 1]
    
    # p60
    beliefs["p60"] = models["p60"].predict_proba(X_part)[:, 1]
    
    # mu_points (conditional on playing)
    X_perf = df[features].fillna(0)
    beliefs["mu_points"] = models["mu_points"].predict(X_perf)
    
    # p_haul (conditional on playing)
    beliefs["p_haul"] = models["p_haul"].predict_proba(X_perf)[:, 1]
    
    # Expected points = p_play * mu_points
    beliefs["expected_points"] = beliefs["p_play"] * beliefs["mu_points"]
    
    return beliefs


# -----------------------------------------------------------------------------
# Decision Evaluation Helpers
# -----------------------------------------------------------------------------


def evaluate_captain_regret(
    beliefs: pd.DataFrame,
    targets: pd.DataFrame,
    gw_range: tuple[int, int],
) -> dict:
    """
    Evaluate captain decision regret.
    
    Decision rule: argmax(expected_points) per GW
    Regret: optimal_points - chosen_points (doubled)
    """
    merged = beliefs.merge(
        targets[["player_id", "gw", "y_points"]],
        on=["player_id", "gw"],
    )
    
    results = []
    
    for gw in range(gw_range[0], gw_range[1] + 1):
        gw_df = merged[merged["gw"] == gw]
        if len(gw_df) == 0:
            continue
        
        # Decision: argmax(expected_points)
        chosen_idx = gw_df["expected_points"].idxmax()
        chosen_points = gw_df.loc[chosen_idx, "y_points"] * 2
        
        # Oracle: argmax(actual_points)
        optimal_idx = gw_df["y_points"].idxmax()
        optimal_points = gw_df.loc[optimal_idx, "y_points"] * 2
        
        regret = optimal_points - chosen_points
        results.append({
            "gw": gw,
            "chosen_points": chosen_points,
            "optimal_points": optimal_points,
            "regret": regret,
        })
    
    if not results:
        return {"mean_regret": float("nan"), "total_regret": 0, "n_gws": 0}
    
    df_results = pd.DataFrame(results)
    return {
        "mean_regret": df_results["regret"].mean(),
        "total_regret": df_results["regret"].sum(),
        "high_regret_rate": (df_results["regret"] >= 10).mean(),
        "n_gws": len(df_results),
    }


def evaluate_transfer_regret(
    beliefs: pd.DataFrame,
    targets: pd.DataFrame,
    gw_range: tuple[int, int],
) -> dict:
    """
    Evaluate transfer-in decision regret.
    
    Decision rule: argmax(expected_points) per GW
    Regret: oracle_points - chosen_points
    """
    merged = beliefs.merge(
        targets[["player_id", "gw", "y_points"]],
        on=["player_id", "gw"],
    )
    
    results = []
    
    for gw in range(gw_range[0], gw_range[1] + 1):
        gw_df = merged[merged["gw"] == gw]
        if len(gw_df) == 0:
            continue
        
        # Decision: argmax(expected_points)
        chosen_idx = gw_df["expected_points"].idxmax()
        chosen_points = gw_df.loc[chosen_idx, "y_points"]
        
        # Oracle: argmax(actual_points)
        optimal_idx = gw_df["y_points"].idxmax()
        optimal_points = gw_df.loc[optimal_idx, "y_points"]
        
        regret = optimal_points - chosen_points
        results.append({
            "gw": gw,
            "chosen_points": chosen_points,
            "optimal_points": optimal_points,
            "regret": regret,
        })
    
    if not results:
        return {"mean_regret": float("nan"), "total_regret": 0, "n_gws": 0}
    
    df_results = pd.DataFrame(results)
    return {
        "mean_regret": df_results["regret"].mean(),
        "total_regret": df_results["regret"].sum(),
        "high_regret_rate": (df_results["regret"] >= 10).mean(),
        "n_gws": len(df_results),
    }


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------


def run_ablation_pipeline(project_root: Path) -> dict:
    """
    Run full ablation pipeline:
    1. Load data
    2. Walk-forward validation (baseline vs fixture_v2)
    3. Train final models for both variants
    4. Generate beliefs
    5. Evaluate decision metrics
    """
    print("=" * 60)
    print("Stage 5c — Belief Models with Fixture Context v2 (Ablation)")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_data(project_root)
    targets = pd.read_csv(
        project_root / "storage" / "research" / "datasets" / "targets.csv"
    )
    print(f"  Loaded {len(df):,} rows")
    
    # Walk-forward validation
    print("\n[2/5] Walk-forward validation...")
    wf_results = walk_forward_ablation(df)
    wf_summary = summarize_ablation(wf_results)
    
    print("\n  Belief-level metrics (walk-forward):")
    print(f"    Baseline:   mu_points MAE = {wf_summary['baseline']['mu_points_mae']:.4f}")
    print(f"    Fixture v2: mu_points MAE = {wf_summary['fixture_v2']['mu_points_mae']:.4f}")
    print(f"    Baseline:   p_haul log_loss = {wf_summary['baseline']['p_haul_log_loss']:.4f}")
    print(f"    Fixture v2: p_haul log_loss = {wf_summary['fixture_v2']['p_haul_log_loss']:.4f}")
    
    # Train final models
    print("\n[3/5] Training final models...")
    models_baseline = train_final_models(df, use_fixture_context_v2=False)
    models_fixture_v2 = train_final_models(df, use_fixture_context_v2=True)
    print("  Trained baseline and fixture_v2 models")
    
    # Generate beliefs
    print("\n[4/5] Generating beliefs...")
    beliefs_baseline = predict_beliefs(df, models_baseline)
    beliefs_fixture_v2 = predict_beliefs(df, models_fixture_v2)
    
    # Save beliefs
    beliefs_dir = project_root / "storage" / "research" / "datasets"
    beliefs_baseline.to_csv(beliefs_dir / "beliefs_fixture_v2_baseline.csv", index=False)
    beliefs_fixture_v2.to_csv(beliefs_dir / "beliefs_fixture_v2_augmented.csv", index=False)
    print(f"  Saved beliefs to {beliefs_dir}")
    
    # Evaluate decision metrics
    print("\n[5/5] Evaluating decision metrics...")
    gw_range = (6, 22)  # GWs with full feature coverage
    
    captain_baseline = evaluate_captain_regret(beliefs_baseline, targets, gw_range)
    captain_fixture_v2 = evaluate_captain_regret(beliefs_fixture_v2, targets, gw_range)
    
    transfer_baseline = evaluate_transfer_regret(beliefs_baseline, targets, gw_range)
    transfer_fixture_v2 = evaluate_transfer_regret(beliefs_fixture_v2, targets, gw_range)
    
    # Compile results
    results = {
        "walk_forward": wf_summary,
        "captain": {
            "baseline": captain_baseline,
            "fixture_v2": captain_fixture_v2,
        },
        "transfer": {
            "baseline": transfer_baseline,
            "fixture_v2": transfer_fixture_v2,
        },
    }
    
    # Print decision metrics
    print("\n  Captain Decision:")
    print(f"    Baseline:   mean_regret = {captain_baseline['mean_regret']:.2f} pts/GW")
    print(f"    Fixture v2: mean_regret = {captain_fixture_v2['mean_regret']:.2f} pts/GW")
    
    print("\n  Transfer-In Decision:")
    print(f"    Baseline:   mean_regret = {transfer_baseline['mean_regret']:.2f} pts/GW")
    print(f"    Fixture v2: mean_regret = {transfer_fixture_v2['mean_regret']:.2f} pts/GW")
    
    # Decision gate
    print("\n" + "=" * 60)
    print("DECISION GATE")
    print("=" * 60)
    
    captain_improvement = captain_baseline["mean_regret"] - captain_fixture_v2["mean_regret"]
    transfer_improvement = transfer_baseline["mean_regret"] - transfer_fixture_v2["mean_regret"]
    
    captain_pass = captain_improvement >= 0
    transfer_pass = transfer_improvement >= 0
    
    print(f"  Captain:  {'✓ PASS' if captain_pass else '✗ FAIL'} (Δ = {captain_improvement:+.2f} pts/GW)")
    print(f"  Transfer: {'✓ PASS' if transfer_pass else '✗ FAIL'} (Δ = {transfer_improvement:+.2f} pts/GW)")
    
    if captain_pass and transfer_pass:
        print("\n  RECOMMENDATION: ACCEPT fixture context v2 features")
        results["recommendation"] = "ACCEPT"
    elif captain_pass or transfer_pass:
        print("\n  RECOMMENDATION: DECISION-SPECIFIC (accept for passing decisions only)")
        results["recommendation"] = "DECISION_SPECIFIC"
    else:
        print("\n  RECOMMENDATION: REJECT fixture context v2 features")
        results["recommendation"] = "REJECT"
    
    print("=" * 60)
    
    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for Stage 5c."""
    project_root = Path(__file__).resolve().parents[4]
    
    results = run_ablation_pipeline(project_root)
    
    # Save results
    reports_dir = project_root / "storage" / "research" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    with open(reports_dir / "fixture_context_v2_ablation.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to {reports_dir / 'fixture_context_v2_ablation.json'}")


if __name__ == "__main__":
    main()
