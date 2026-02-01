"""
Stage 5b — Belief Model Retraining with Fixture Context (Ablation)

Controlled ablation experiment to test whether adding `opp_def_strength`
improves belief quality and downstream decision performance.

Ablation Design:
    - Participation models (p_play, p60): UNCHANGED (no fixture features)
    - Performance models (mu_points, p_haul): ADD opp_def_strength

This isolates the effect of fixture context on performance predictions.

Inputs:
    storage/datasets/targets.csv
    storage/datasets/features_participation.csv
    storage/datasets/features_performance.csv
    storage/datasets/features_fixture_context.csv

Outputs:
    storage/datasets/beliefs_fixture_context.csv
    storage/models/fixture_context/*.pkl

Evaluation:
    - Belief-level metrics (log-loss, MAE)
    - Decision-level regret (captain policy)
"""

import sqlite3
from pathlib import Path
from typing import Any

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

# Original performance features
PERFORMANCE_FEATURES_BASELINE = [
    "points_per_90_5",
    "xGI_per_90_5",
    "bonus_per_90_5",
    "ict_per_90_5",
]

# Augmented performance features (with fixture context)
PERFORMANCE_FEATURES_AUGMENTED = PERFORMANCE_FEATURES_BASELINE + [
    "opp_def_strength",
]

# Walk-forward fold boundaries (same as Stage 5)
WALK_FORWARD_FOLDS = [
    {"train_end": 5, "val_start": 6, "val_end": 8}, 
    {"train_end": 8, "val_start": 9, "val_end": 11},   
    {"train_end": 11, "val_start": 12, "val_end": 14},  
    {"train_end": 14, "val_start": 15, "val_end": 17}, 
    {"train_end": 17, "val_start": 18, "val_end": 20},  
    {"train_end": 20, "val_start": 21, "val_end": 22},  
]

FINAL_TRAIN_END = 22

# LightGBM hyperparameters (same as Stage 5)
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
    Load and join targets with all feature tables.
    
    Includes fixture context features for ablation.
    """
    targets_path = project_root / "storage" / "research" / "datasets" / "targets.csv"
    participation_path = project_root / "storage" / "research" / "datasets" / "features_participation.csv"
    performance_path = project_root / "storage" / "research" / "datasets" / "features_performance.csv"
    fixture_path = project_root / "storage" / "research" / "datasets" / "features_fixture_context.csv"

    targets = pd.read_csv(targets_path)
    participation = pd.read_csv(participation_path)
    performance = pd.read_csv(performance_path)
    fixture = pd.read_csv(fixture_path)

    # Join on (player_id, gw)
    df = targets.merge(participation, on=["player_id", "gw"], how="left")
    df = df.merge(performance, on=["player_id", "gw"], how="left")
    df = df.merge(fixture, on=["player_id", "gw"], how="left")

    return df


# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------


def prepare_participation_data(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare data for participation models (unchanged from Stage 5)."""
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
    
    Args:
        df: Data with features and targets
        target_col: Target column name
        features: List of feature columns to use
    """
    played_mask = df["y_play"] == 1
    df_played = df[played_mask].copy()

    X = df_played[features].copy()
    y = df_played[target_col].copy()

    # Drop rows with NaN features
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

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
    """Train participation model (same as Stage 5)."""
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
    Run walk-forward validation comparing baseline vs augmented features.
    
    Returns metrics for both variants.
    """
    results = {
        "baseline": {"mu_points": [], "p_haul": []},
        "augmented": {"mu_points": [], "p_haul": []},
    }
    
    for fold in WALK_FORWARD_FOLDS:
        df_train, df_val = split_by_gw(
            df, fold["train_end"], fold["val_start"], fold["val_end"]
        )
        fold_label = f"GW {fold['val_start']}-{fold['val_end']}"
        
        # --- mu_points: Baseline ---
        model_base = train_performance_model(
            df_train, "y_points", PERFORMANCE_FEATURES_BASELINE, is_classifier=False
        )
        X_val, y_val, val_idx = prepare_performance_data(
            df_val, "y_points", PERFORMANCE_FEATURES_BASELINE
        )
        if len(X_val) > 0:
            y_pred = model_base.predict(X_val)
            metrics = compute_regression_metrics(y_val.values, y_pred)
            metrics["fold"] = fold_label
            results["baseline"]["mu_points"].append(metrics)
        
        # --- mu_points: Augmented ---
        model_aug = train_performance_model(
            df_train, "y_points", PERFORMANCE_FEATURES_AUGMENTED, is_classifier=False
        )
        X_val_aug, y_val_aug, _ = prepare_performance_data(
            df_val, "y_points", PERFORMANCE_FEATURES_AUGMENTED
        )
        if len(X_val_aug) > 0:
            y_pred = model_aug.predict(X_val_aug)
            metrics = compute_regression_metrics(y_val_aug.values, y_pred)
            metrics["fold"] = fold_label
            results["augmented"]["mu_points"].append(metrics)
        
        # --- p_haul: Baseline ---
        model_base = train_performance_model(
            df_train, "y_haul", PERFORMANCE_FEATURES_BASELINE, is_classifier=True
        )
        X_val, y_val, _ = prepare_performance_data(
            df_val, "y_haul", PERFORMANCE_FEATURES_BASELINE
        )
        if len(X_val) > 0:
            y_pred = model_base.predict_proba(X_val)[:, 1]
            metrics = compute_classification_metrics(y_val.values, y_pred)
            metrics["fold"] = fold_label
            results["baseline"]["p_haul"].append(metrics)
        
        # --- p_haul: Augmented ---
        model_aug = train_performance_model(
            df_train, "y_haul", PERFORMANCE_FEATURES_AUGMENTED, is_classifier=True
        )
        X_val_aug, y_val_aug, _ = prepare_performance_data(
            df_val, "y_haul", PERFORMANCE_FEATURES_AUGMENTED
        )
        if len(X_val_aug) > 0:
            y_pred = model_aug.predict_proba(X_val_aug)[:, 1]
            metrics = compute_classification_metrics(y_val_aug.values, y_pred)
            metrics["fold"] = fold_label
            results["augmented"]["p_haul"].append(metrics)
    
    return results


# -----------------------------------------------------------------------------
# Final Model Training
# -----------------------------------------------------------------------------


def train_final_models(df: pd.DataFrame) -> dict[str, Any]:
    """
    Train final models with fixture-augmented features.
    
    Participation models: unchanged
    Performance models: use augmented features
    """
    df_train = df[df["gw"] <= FINAL_TRAIN_END]
    
    models = {}
    
    # Participation models (unchanged)
    models["p_play"] = train_participation_model(df_train, "y_play")
    models["p60"] = train_participation_model(df_train, "y_60")
    
    # Performance models (augmented)
    models["mu_points"] = train_performance_model(
        df_train, "y_points", PERFORMANCE_FEATURES_AUGMENTED, is_classifier=False
    )
    models["p_haul"] = train_performance_model(
        df_train, "y_haul", PERFORMANCE_FEATURES_AUGMENTED, is_classifier=True
    )
    
    return models


def predict_beliefs(df: pd.DataFrame, models: dict[str, Any]) -> pd.DataFrame:
    """Generate belief predictions with fixture-augmented models."""
    beliefs = pd.DataFrame({
        "player_id": df["player_id"],
        "gw": df["gw"],
    })
    
    # p_play
    X, _ = prepare_participation_data(df, "y_play")
    beliefs["p_play"] = models["p_play"].predict_proba(X)[:, 1]
    
    # p60
    X, _ = prepare_participation_data(df, "y_60")
    beliefs["p60"] = models["p60"].predict_proba(X)[:, 1]
    
    # mu_points (with augmented features)
    X_perf = df[PERFORMANCE_FEATURES_AUGMENTED].copy()
    valid_mask = X_perf.notna().all(axis=1)
    beliefs["mu_points"] = np.nan
    if valid_mask.sum() > 0:
        beliefs.loc[valid_mask, "mu_points"] = models["mu_points"].predict(X_perf[valid_mask])
    
    # p_haul (with augmented features)
    beliefs["p_haul"] = np.nan
    if valid_mask.sum() > 0:
        beliefs.loc[valid_mask, "p_haul"] = models["p_haul"].predict_proba(X_perf[valid_mask])[:, 1]
    
    # Enforce p60 <= p_play
    violations = beliefs["p60"] > beliefs["p_play"]
    if violations.sum() > 0:
        beliefs.loc[violations, "p60"] = beliefs.loc[violations, "p_play"]
    
    return beliefs


# -----------------------------------------------------------------------------
# Captain Policy Evaluation (Decision-Level)
# -----------------------------------------------------------------------------


def evaluate_captain_regret(
    beliefs: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict:
    """
    Evaluate captain policy using mu_points only (revised policy from Stage 6c).
    
    Returns regret metrics.
    """
    # Compute scores (mu_points only)
    scores = beliefs[["player_id", "gw"]].copy()
    scores["score"] = beliefs["mu_points"].fillna(0)
    
    # Select captain per GW
    idx_max = scores.groupby("gw")["score"].idxmax()
    chosen = scores.loc[idx_max, ["gw", "player_id"]].copy()
    chosen = chosen.rename(columns={"player_id": "chosen_player_id"})
    
    # Get chosen player's actual points
    chosen = chosen.merge(
        targets[["player_id", "gw", "y_points"]].rename(
            columns={"player_id": "chosen_player_id", "y_points": "chosen_points"}
        ),
        on=["gw", "chosen_player_id"],
        how="left",
    )
    
    # Find optimal per GW
    targets_played = targets[targets["y_play"] == 1]
    optimal = (
        targets_played.loc[targets_played.groupby("gw")["y_points"].idxmax()]
        [["gw", "y_points"]]
        .rename(columns={"y_points": "optimal_points"})
    )
    
    # Merge and compute regret
    evaluation = chosen.merge(optimal, on="gw", how="left")
    evaluation["chosen_points"] = evaluation["chosen_points"].fillna(0)
    evaluation["optimal_points"] = evaluation["optimal_points"].fillna(0)
    evaluation["regret"] = evaluation["optimal_points"] - evaluation["chosen_points"]
    
    return {
        "mean_regret": evaluation["regret"].mean(),
        "median_regret": evaluation["regret"].median(),
        "total_regret": evaluation["regret"].sum(),
        "pct_high_regret": (evaluation["regret"] >= 10).mean(),
        "n_gw": len(evaluation),
    }


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    """Stage 5b pipeline entry point."""
    project_root = Path(__file__).resolve().parents[4]
    
    print("=" * 70)
    print("Stage 5b — Belief Model Retraining with Fixture Context (Ablation)")
    print("=" * 70)
    
    # 1. Load data
    print("\n1. Loading data...")
    df = load_data(project_root)
    print(f"   Rows: {len(df):,}")
    print(f"   Players: {df['player_id'].nunique():,}")
    print(f"   GWs: {df['gw'].min()} to {df['gw'].max()}")
    
    # Check fixture feature coverage
    gw2_plus = df[df["gw"] >= 2]
    coverage = gw2_plus["opp_def_strength"].notna().mean()
    print(f"   opp_def_strength coverage (GW >= 2): {coverage:.1%}")
    
    # 2. Walk-forward ablation
    print("\n2. Running walk-forward ablation...")
    ablation_results = walk_forward_ablation(df)
    
    # 3. Report belief-level metrics
    print("\n" + "=" * 70)
    print("Belief-Level Metrics: Baseline vs Fixture-Augmented")
    print("=" * 70)
    
    # mu_points (MAE)
    print("\n### mu_points (MAE)")
    print(f"{'Fold':<15} {'Baseline':>12} {'Augmented':>12} {'Δ':>10}")
    print("-" * 50)
    
    for i, (b, a) in enumerate(zip(
        ablation_results["baseline"]["mu_points"],
        ablation_results["augmented"]["mu_points"],
    )):
        delta = a["mae"] - b["mae"]
        print(f"{b['fold']:<15} {b['mae']:>12.3f} {a['mae']:>12.3f} {delta:>+10.3f}")
    
    avg_base = np.mean([m["mae"] for m in ablation_results["baseline"]["mu_points"]])
    avg_aug = np.mean([m["mae"] for m in ablation_results["augmented"]["mu_points"]])
    print("-" * 50)
    print(f"{'Average':<15} {avg_base:>12.3f} {avg_aug:>12.3f} {avg_aug - avg_base:>+10.3f}")
    
    # p_haul (log loss)
    print("\n### p_haul (Log Loss)")
    print(f"{'Fold':<15} {'Baseline':>12} {'Augmented':>12} {'Δ':>10}")
    print("-" * 50)
    
    for i, (b, a) in enumerate(zip(
        ablation_results["baseline"]["p_haul"],
        ablation_results["augmented"]["p_haul"],
    )):
        delta = a["log_loss"] - b["log_loss"]
        print(f"{b['fold']:<15} {b['log_loss']:>12.4f} {a['log_loss']:>12.4f} {delta:>+10.4f}")
    
    avg_base = np.mean([m["log_loss"] for m in ablation_results["baseline"]["p_haul"]])
    avg_aug = np.mean([m["log_loss"] for m in ablation_results["augmented"]["p_haul"]])
    print("-" * 50)
    print(f"{'Average':<15} {avg_base:>12.4f} {avg_aug:>12.4f} {avg_aug - avg_base:>+10.4f}")
    
    # 4. Train final models
    print("\n3. Training final models with fixture context...")
    models = train_final_models(df)
    
    # 5. Generate beliefs
    print("4. Generating beliefs...")
    beliefs = predict_beliefs(df, models)
    
    # 6. Evaluate captain regret
    print("\n" + "=" * 70)
    print("Decision-Level Metrics: Captain Regret")
    print("=" * 70)
    
    targets = pd.read_csv(project_root / "storage" / "research" / "datasets" / "targets.csv")
    
    # Baseline beliefs (original Stage 5)
    baseline_beliefs = pd.read_csv(project_root / "storage" / "research" / "datasets" / "beliefs.csv")
    baseline_regret = evaluate_captain_regret(baseline_beliefs, targets)
    
    # Augmented beliefs
    augmented_regret = evaluate_captain_regret(beliefs, targets)
    
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Augmented':>12} {'Δ':>10}")
    print("-" * 60)
    print(f"{'Mean regret':<25} {baseline_regret['mean_regret']:>12.2f} {augmented_regret['mean_regret']:>12.2f} {augmented_regret['mean_regret'] - baseline_regret['mean_regret']:>+10.2f}")
    print(f"{'Median regret':<25} {baseline_regret['median_regret']:>12.2f} {augmented_regret['median_regret']:>12.2f} {augmented_regret['median_regret'] - baseline_regret['median_regret']:>+10.2f}")
    print(f"{'% GW regret >= 10':<25} {baseline_regret['pct_high_regret']:>12.1%} {augmented_regret['pct_high_regret']:>12.1%}")
    print(f"{'Total regret':<25} {baseline_regret['total_regret']:>12.0f} {augmented_regret['total_regret']:>12.0f} {augmented_regret['total_regret'] - baseline_regret['total_regret']:>+10.0f}")
    
    # 7. Save outputs
    print("\n5. Saving outputs...")
    
    # Save beliefs
    beliefs_path = project_root / "storage" / "research" / "datasets" / "beliefs_fixture_context.csv"
    beliefs.to_csv(beliefs_path, index=False)
    print(f"   Beliefs: {beliefs_path}")
    
    # Save models
    models_dir = project_root / "storage" / "research" / "models" / "fixture_context"
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model_path = models_dir / f"{name}.pkl"
        joblib.dump(model, model_path)
    print(f"   Models: {models_dir}")
    
    # 8. Conclusion
    print("\n" + "=" * 70)
    print("Stage 5b Complete — Ablation Summary")
    print("=" * 70)
    
    mu_delta = avg_aug - np.mean([m["mae"] for m in ablation_results["baseline"]["mu_points"]])
    regret_delta = augmented_regret['mean_regret'] - baseline_regret['mean_regret']
    
    if mu_delta < 0 and regret_delta < 0:
        print("\n✅ Fixture context IMPROVES both belief quality and decision performance.")
        print("   Recommendation: Freeze augmented beliefs for production.")
    elif mu_delta < 0 and regret_delta >= 0:
        print("\n⚠️  Fixture context improves belief MAE but NOT decision regret.")
        print("   Recommendation: Do NOT freeze. Better beliefs ≠ better decisions.")
    elif mu_delta >= 0 and regret_delta < 0:
        print("\n⚠️  Fixture context degrades belief MAE but IMPROVES decision regret.")
        print("   Recommendation: Investigate further. Unexpected correlation.")
    else:
        print("\n❌ Fixture context does NOT improve performance.")
        print("   Recommendation: Do NOT freeze. Keep baseline beliefs.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
