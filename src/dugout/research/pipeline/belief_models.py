"""
Stage 5 — Instrumental Belief Models

Trains belief estimators (probabilities/expectations) from frozen targets and features.
Produces beliefs, not decisions or rankings.

Models:
    p_play     P(minutes > 0)              trained on all rows
    p60        P(minutes >= 60)            trained on all rows
    mu_points  E[points | plays]           trained on rows where y_play==1
    p_haul     P(points >= 10 | plays)     trained on rows where y_play==1

Outputs:
    storage/models/*.pkl                   Fitted model artifacts
    storage/datasets/beliefs.csv           Scored beliefs for all (player_id, gw)
    storage/reports/stage5_belief_models.md  Evaluation report

Contract:
    - Features from GWs <= t-1 only (enforced by Stage 3/4a)
    - Time-aware splits (walk-forward, no shuffling)
    - No decision logic, ranking, or recommendations
"""

import json
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

PERFORMANCE_FEATURES = [
    "points_per_90_5",
    "xGI_per_90_5",
    "bonus_per_90_5",
    "ict_per_90_5",
]

# Walk-forward fold boundaries (train end GW, val start GW, val end GW)
# Train on GWs 1..train_end, validate on val_start..val_end
# First validation starts at GW 6 (first GW with full 5-GW feature window)
WALK_FORWARD_FOLDS = [
    {"train_end": 5, "val_start": 6, "val_end": 8}, 
    {"train_end": 8, "val_start": 9, "val_end": 11},   
    {"train_end": 11, "val_start": 12, "val_end": 14},  
    {"train_end": 14, "val_start": 15, "val_end": 17}, 
    {"train_end": 17, "val_start": 18, "val_end": 20},  
    {"train_end": 20, "val_start": 21, "val_end": 22},  
]

# Final training uses all available data for deployment
FINAL_TRAIN_END = 22

# LightGBM hyperparameters (simple, low depth)
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
    Load and join targets with participation and performance features.

    Returns a single DataFrame aligned on (player_id, gw).
    """
    targets_path = project_root / "storage" / "research" / "datasets" / "targets.csv"
    participation_path = project_root / "storage" / "research" / "datasets" / "features_participation.csv"
    performance_path = project_root / "storage" / "research" / "datasets" / "features_performance.csv"

    targets = pd.read_csv(targets_path)
    participation = pd.read_csv(participation_path)
    performance = pd.read_csv(performance_path)

    # Join on (player_id, gw)
    df = targets.merge(participation, on=["player_id", "gw"], how="left")
    df = df.merge(performance, on=["player_id", "gw"], how="left")

    return df


# -----------------------------------------------------------------------------
# Baseline Computation
# -----------------------------------------------------------------------------


def compute_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute baseline predictions from Stage 3/4a features.

    Baselines:
        p_play_baseline = p_play_hat (Stage 3)
        p60_baseline = p60_hat (Stage 3)
        mu_points_baseline = points_per_90_5 (Stage 4a, conditional intensity)
        p_haul_baseline = global training haul rate (among players who played)
    """
    df = df.copy()

    # Direct feature mappings for participation baselines
    df["p_play_baseline"] = df["p_play_hat"]
    df["p60_baseline"] = df["p60_hat"]

    # Conditional performance baseline
    df["mu_points_baseline"] = df["points_per_90_5"]

    # Haul baseline: use global rate from training data (y_play==1 rows)
    # This is computed later during training; placeholder for now
    df["p_haul_baseline"] = np.nan

    return df


# -----------------------------------------------------------------------------
# Missingness Handling
# -----------------------------------------------------------------------------


def prepare_participation_data(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for participation models (p_play, p60).

    Imputes missing participation features with 0 (first GW per player).
    """
    X = df[PARTICIPATION_FEATURES].copy()
    y = df[target_col].copy()

    # Impute NaNs with 0 (player's first GW has no history)
    X = X.fillna(0)

    return X, y


def prepare_performance_data(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    """
    Prepare data for conditional performance models (mu_points, p_haul).

    Filters to y_play==1 rows and drops rows with missing performance features.
    Returns the valid index for alignment.
    """
    # Filter to rows where player actually played
    played_mask = df["y_play"] == 1
    df_played = df[played_mask].copy()

    X = df_played[PERFORMANCE_FEATURES].copy()
    y = df_played[target_col].copy()

    # Drop rows with NaN features (no prior appearances)
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y, X.index


# -----------------------------------------------------------------------------
# Time-Aware Splitting
# -----------------------------------------------------------------------------


def split_by_gw(
    df: pd.DataFrame,
    train_end: int,
    val_start: int,
    val_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by gameweek for walk-forward validation.

    Train: GWs 1..train_end
    Val: GWs val_start..val_end
    """
    train_mask = df["gw"] <= train_end
    val_mask = (df["gw"] >= val_start) & (df["gw"] <= val_end)

    return df[train_mask], df[val_mask]


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    name: str,
) -> dict:
    """Compute log loss and Brier score for probability predictions."""
    # Clip to avoid log(0)
    y_pred_clipped = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)

    return {
        "model": name,
        "log_loss": log_loss(y_true, y_pred_clipped),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
        "n_samples": len(y_true),
        "positive_rate": y_true.mean(),
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
) -> dict:
    """Compute MAE and bias for regression predictions."""
    return {
        "model": name,
        "mae": mean_absolute_error(y_true, y_pred),
        "bias": (y_pred - y_true).mean(),
        "n_samples": len(y_true),
    }


def compute_calibration_table(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Compute calibration table for probability predictions.

    Returns predicted vs observed rates for each bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            rows.append({
                "bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                "n": mask.sum(),
                "mean_predicted": y_pred_proba[mask].mean(),
                "mean_observed": y_true[mask].mean(),
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Model Training
# -----------------------------------------------------------------------------


def train_participation_model(
    df_train: pd.DataFrame,
    target_col: str,
    model_class: type = LGBMClassifier,
) -> tuple[Any, dict]:
    """Train a participation model (p_play or p60)."""
    X_train, y_train = prepare_participation_data(df_train, target_col)

    model = model_class(**LGBM_CLASSIFIER_PARAMS)
    model.fit(X_train, y_train)

    metadata = {
        "target": target_col,
        "features": PARTICIPATION_FEATURES,
        "train_gw_range": (df_train["gw"].min(), df_train["gw"].max()),
        "n_train": len(X_train),
    }

    return model, metadata


def train_performance_model(
    df_train: pd.DataFrame,
    target_col: str,
    model_class: type,
    is_classifier: bool = False,
) -> tuple[Any, dict]:
    """Train a conditional performance model (mu_points or p_haul)."""
    X_train, y_train, _ = prepare_performance_data(df_train, target_col)

    if is_classifier:
        model = model_class(**LGBM_CLASSIFIER_PARAMS)
    else:
        model = model_class(**LGBM_REGRESSOR_PARAMS)

    model.fit(X_train, y_train)

    metadata = {
        "target": target_col,
        "features": PERFORMANCE_FEATURES,
        "train_gw_range": (df_train["gw"].min(), df_train["gw"].max()),
        "n_train": len(X_train),
        "n_dropped_missing": len(df_train[df_train["y_play"] == 1]) - len(X_train),
    }

    return model, metadata


# -----------------------------------------------------------------------------
# Walk-Forward Validation
# -----------------------------------------------------------------------------


def walk_forward_validate(
    df: pd.DataFrame,
    folds: list[dict],
) -> dict:
    """
    Perform walk-forward validation for all four models.

    Returns metrics per fold and aggregated.
    """
    results = {
        "p_play": {"model": [], "baseline": []},
        "p60": {"model": [], "baseline": []},
        "mu_points": {"model": [], "baseline": []},
        "p_haul": {"model": [], "baseline": []},
    }

    for fold in folds:
        df_train, df_val = split_by_gw(
            df, fold["train_end"], fold["val_start"], fold["val_end"]
        )

        fold_label = f"GW {fold['val_start']}-{fold['val_end']}"

        # --- p_play ---
        model, _ = train_participation_model(df_train, "y_play")
        X_val, y_val = prepare_participation_data(df_val, "y_play")
        y_pred = model.predict_proba(X_val)[:, 1]
        results["p_play"]["model"].append(
            compute_classification_metrics(y_val.values, y_pred, fold_label)
        )
        # Baseline
        baseline = df_val["p_play_hat"].fillna(0).values
        results["p_play"]["baseline"].append(
            compute_classification_metrics(y_val.values, baseline, fold_label)
        )

        # --- p60 ---
        model, _ = train_participation_model(df_train, "y_60")
        X_val, y_val = prepare_participation_data(df_val, "y_60")
        y_pred = model.predict_proba(X_val)[:, 1]
        results["p60"]["model"].append(
            compute_classification_metrics(y_val.values, y_pred, fold_label)
        )
        # Baseline
        baseline = df_val["p60_hat"].fillna(0).values
        results["p60"]["baseline"].append(
            compute_classification_metrics(y_val.values, baseline, fold_label)
        )

        # --- mu_points (conditional on playing) ---
        model, _ = train_performance_model(
            df_train, "y_points", LGBMRegressor, is_classifier=False
        )
        X_val, y_val, val_idx = prepare_performance_data(df_val, "y_points")
        if len(X_val) > 0:
            y_pred = model.predict(X_val)
            results["mu_points"]["model"].append(
                compute_regression_metrics(y_val.values, y_pred, fold_label)
            )
            # Baseline: points_per_90_5
            baseline = df_val.loc[val_idx, "points_per_90_5"].values
            valid_baseline = ~np.isnan(baseline)
            if valid_baseline.sum() > 0:
                results["mu_points"]["baseline"].append(
                    compute_regression_metrics(
                        y_val.values[valid_baseline],
                        baseline[valid_baseline],
                        fold_label,
                    )
                )

        # --- p_haul (conditional on playing) ---
        model, _ = train_performance_model(
            df_train, "y_haul", LGBMClassifier, is_classifier=True
        )
        X_val, y_val, val_idx = prepare_performance_data(df_val, "y_haul")
        if len(X_val) > 0:
            y_pred = model.predict_proba(X_val)[:, 1]
            results["p_haul"]["model"].append(
                compute_classification_metrics(y_val.values, y_pred, fold_label)
            )
            # Baseline: global haul rate from training set
            train_played = df_train[df_train["y_play"] == 1]
            haul_rate = train_played["y_haul"].mean()
            baseline = np.full(len(y_val), haul_rate)
            results["p_haul"]["baseline"].append(
                compute_classification_metrics(y_val.values, baseline, fold_label)
            )

    return results


# -----------------------------------------------------------------------------
# Final Model Training & Prediction
# -----------------------------------------------------------------------------


def train_final_models(
    df: pd.DataFrame,
    train_end: int,
) -> dict[str, tuple[Any, dict]]:
    """
    Train final models on all data up to train_end for deployment.

    Returns dict mapping model name to (model, metadata).
    """
    df_train = df[df["gw"] <= train_end]

    models = {}

    # p_play
    model, meta = train_participation_model(df_train, "y_play")
    models["p_play"] = (model, meta)

    # p60
    model, meta = train_participation_model(df_train, "y_60")
    models["p60"] = (model, meta)

    # mu_points
    model, meta = train_performance_model(
        df_train, "y_points", LGBMRegressor, is_classifier=False
    )
    models["mu_points"] = (model, meta)

    # p_haul
    model, meta = train_performance_model(
        df_train, "y_haul", LGBMClassifier, is_classifier=True
    )
    # Store global haul rate for baseline
    train_played = df_train[df_train["y_play"] == 1]
    meta["global_haul_rate"] = train_played["y_haul"].mean()
    models["p_haul"] = (model, meta)

    return models


def predict_beliefs(
    df: pd.DataFrame,
    models: dict[str, tuple[Any, dict]],
) -> pd.DataFrame:
    """
    Generate belief predictions for all (player_id, gw) rows.

    Returns DataFrame with player_id, gw, p_play, p60, mu_points, p_haul.
    """
    beliefs = pd.DataFrame({
        "player_id": df["player_id"],
        "gw": df["gw"],
    })

    # --- p_play ---
    model, _ = models["p_play"]
    X, _ = prepare_participation_data(df, "y_play")
    beliefs["p_play"] = model.predict_proba(X)[:, 1]

    # --- p60 ---
    model, _ = models["p60"]
    X, _ = prepare_participation_data(df, "y_60")
    beliefs["p60"] = model.predict_proba(X)[:, 1]

    # --- mu_points (conditional, but predict for all rows) ---
    model, _ = models["mu_points"]
    X_perf = df[PERFORMANCE_FEATURES].copy()
    # For rows with missing features, predict NaN
    valid_mask = X_perf.notna().all(axis=1)
    beliefs["mu_points"] = np.nan
    if valid_mask.sum() > 0:
        beliefs.loc[valid_mask, "mu_points"] = model.predict(X_perf[valid_mask])

    # --- p_haul (conditional, but predict for all rows) ---
    model, meta = models["p_haul"]
    beliefs["p_haul"] = np.nan
    if valid_mask.sum() > 0:
        beliefs.loc[valid_mask, "p_haul"] = model.predict_proba(X_perf[valid_mask])[:, 1]

    # --- Post-hoc enforcement: p60 <= p_play ---
    # Independent models can produce violations; clip to maintain logical consistency
    violations = beliefs["p60"] > beliefs["p_play"]
    if violations.sum() > 0:
        beliefs.loc[violations, "p60"] = beliefs.loc[violations, "p_play"]

    return beliefs


# -----------------------------------------------------------------------------
# Quality Checks
# -----------------------------------------------------------------------------


def enforce_belief_contract(beliefs: pd.DataFrame, targets: pd.DataFrame) -> None:
    """
    Enforce Stage 5 contract on belief outputs.

    Raises AssertionError if any check fails.
    """
    # Check 1: Row count matches targets
    assert len(beliefs) == len(targets), (
        f"Belief row count ({len(beliefs)}) != target row count ({len(targets)})"
    )

    # Check 2: No duplicate (player_id, gw)
    duplicates = beliefs.duplicated(subset=["player_id", "gw"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (player_id, gw) pairs"

    # Check 3: Probability bounds [0, 1]
    for col in ["p_play", "p60", "p_haul"]:
        valid = beliefs[col].dropna()
        assert (valid >= 0).all() and (valid <= 1).all(), (
            f"{col} has values outside [0, 1]"
        )

    # Check 4: mu_points is finite where not NaN
    valid_mu = beliefs["mu_points"].dropna()
    assert np.isfinite(valid_mu).all(), "mu_points has non-finite values"

    # Check 5: Report p60 > p_play violations (do not fail, just report)
    violations = (beliefs["p60"] > beliefs["p_play"] + 1e-6).sum()
    if violations > 0:
        print(f"WARNING: {violations} rows where p60 > p_play")


# -----------------------------------------------------------------------------
# Validation Results Storage
# -----------------------------------------------------------------------------


def save_validation_results(
    cv_results: dict,
    folds: list[dict],
    output_path: Path,
) -> None:
    """
    Save walk-forward validation results as JSON for reuse.
    
    Structure:
    {
        "folds": [...],
        "models": {
            "p_play": {"model": [...], "baseline": [...]},
            ...
        }
    }
    """
    data = {
        "folds": folds,
        "models": cv_results,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_validation_results(input_path: Path) -> dict:
    """Load walk-forward validation results from JSON."""
    with open(input_path, "r") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------------


def generate_report(
    cv_results: dict,
    models: dict[str, tuple[Any, dict]],
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate markdown evaluation report."""
    lines = [
        "# Stage 5 — Belief Models Evaluation Report",
        "",
        "## Overview",
        "",
        "This report summarizes the training and evaluation of instrumental belief models.",
        "",
        "### Models Trained",
        "",
        "| Model | Target | Training Subset | Features |",
        "|-------|--------|-----------------|----------|",
        "| p_play | y_play | all rows | participation |",
        "| p60 | y_60 | all rows | participation |",
        "| mu_points | y_points | y_play==1 only | performance |",
        "| p_haul | y_haul | y_play==1 only | performance |",
        "",
        "---",
        "",
        "## Walk-Forward Validation",
        "",
        "### Split Strategy",
        "",
        "Time-aware walk-forward validation with the following folds:",
        "",
    ]

    for fold in WALK_FORWARD_FOLDS:
        lines.append(
            f"- Train: GW 1-{fold['train_end']}, Validate: GW {fold['val_start']}-{fold['val_end']}"
        )

    lines.extend(["", "---", ""])

    # Classification model results
    for model_name in ["p_play", "p60", "p_haul"]:
        lines.extend([
            f"### {model_name}",
            "",
            "#### Model vs Baseline (Log Loss / Brier Score)",
            "",
            "| Fold | Model LL | Baseline LL | Model Brier | Baseline Brier |",
            "|------|----------|-------------|-------------|----------------|",
        ])

        for i, (m, b) in enumerate(zip(
            cv_results[model_name]["model"],
            cv_results[model_name]["baseline"],
        )):
            lines.append(
                f"| {m['model']} | {m['log_loss']:.4f} | {b['log_loss']:.4f} | "
                f"{m['brier_score']:.4f} | {b['brier_score']:.4f} |"
            )

        # Aggregate
        if cv_results[model_name]["model"]:
            avg_model_ll = np.mean([m["log_loss"] for m in cv_results[model_name]["model"]])
            avg_base_ll = np.mean([b["log_loss"] for b in cv_results[model_name]["baseline"]])
            avg_model_br = np.mean([m["brier_score"] for m in cv_results[model_name]["model"]])
            avg_base_br = np.mean([b["brier_score"] for b in cv_results[model_name]["baseline"]])
            lines.append(
                f"| **Average** | **{avg_model_ll:.4f}** | **{avg_base_ll:.4f}** | "
                f"**{avg_model_br:.4f}** | **{avg_base_br:.4f}** |"
            )

        lines.extend(["", ""])

    # Regression model results
    lines.extend([
        "### mu_points",
        "",
        "#### Model vs Baseline (MAE / Bias)",
        "",
        "| Fold | Model MAE | Baseline MAE | Model Bias | Baseline Bias |",
        "|------|-----------|--------------|------------|---------------|",
    ])

    for i, m in enumerate(cv_results["mu_points"]["model"]):
        b = cv_results["mu_points"]["baseline"][i] if i < len(cv_results["mu_points"]["baseline"]) else {"mae": np.nan, "bias": np.nan}
        lines.append(
            f"| {m['model']} | {m['mae']:.2f} | {b['mae']:.2f} | "
            f"{m['bias']:.2f} | {b['bias']:.2f} |"
        )

    if cv_results["mu_points"]["model"]:
        avg_model_mae = np.mean([m["mae"] for m in cv_results["mu_points"]["model"]])
        avg_base_mae = np.mean([b["mae"] for b in cv_results["mu_points"]["baseline"]])
        avg_model_bias = np.mean([m["bias"] for m in cv_results["mu_points"]["model"]])
        avg_base_bias = np.mean([b["bias"] for b in cv_results["mu_points"]["baseline"]])
        lines.append(
            f"| **Average** | **{avg_model_mae:.2f}** | **{avg_base_mae:.2f}** | "
            f"**{avg_model_bias:.2f}** | **{avg_base_bias:.2f}** |"
        )

    lines.extend(["", "---", ""])

    # Final model metadata
    lines.extend([
        "## Final Model Training",
        "",
        f"Final models trained on GW 1-{FINAL_TRAIN_END}.",
        "",
    ])

    for name, (model, meta) in models.items():
        lines.extend([
            f"### {name}",
            f"- Target: `{meta['target']}`",
            f"- Features: {meta['features']}",
            f"- Train GW range: {meta['train_gw_range']}",
            f"- Training samples: {meta['n_train']}",
        ])
        if "n_dropped_missing" in meta:
            lines.append(f"- Dropped (missing features): {meta['n_dropped_missing']}")
        if "global_haul_rate" in meta:
            lines.append(f"- Global haul rate (baseline): {meta['global_haul_rate']:.3f}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Missingness Summary",
        "",
    ])

    # Compute missingness stats
    n_total = len(df)
    n_part_missing = df[PARTICIPATION_FEATURES].isna().any(axis=1).sum()
    n_perf_missing = df[PERFORMANCE_FEATURES].isna().any(axis=1).sum()
    n_played = (df["y_play"] == 1).sum()
    n_perf_missing_played = df[df["y_play"] == 1][PERFORMANCE_FEATURES].isna().any(axis=1).sum()

    lines.extend([
        f"- Total rows: {n_total:,}",
        f"- Participation features missing: {n_part_missing:,} ({n_part_missing/n_total:.1%})",
        f"- Performance features missing: {n_perf_missing:,} ({n_perf_missing/n_total:.1%})",
        f"- Rows with y_play==1: {n_played:,}",
        f"- Performance features missing (among y_play==1): {n_perf_missing_played:,} ({n_perf_missing_played/n_played:.1%})",
        "",
        "**Handling:**",
        "- Participation features: imputed with 0 (first GW per player)",
        "- Performance features: dropped rows with missing (no prior appearances)",
        "",
    ])

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


# -----------------------------------------------------------------------------
# Pipeline Entry Point
# -----------------------------------------------------------------------------


def build_belief_models(project_root: Path) -> None:
    """
    Stage 5 pipeline entry point.

    Loads data, performs walk-forward validation, trains final models,
    generates beliefs.csv, and writes evaluation report.
    """
    print("Stage 5 — Instrumental Belief Models")
    print("=" * 50)

    # Load data
    print("\n1. Loading data...")
    df = load_data(project_root)
    df = compute_baselines(df)
    print(f"   Loaded {len(df):,} rows, {df['player_id'].nunique()} players, GW {df['gw'].min()}-{df['gw'].max()}")

    # Walk-forward validation
    print("\n2. Walk-forward validation...")
    cv_results = walk_forward_validate(df, WALK_FORWARD_FOLDS)
    print(f"   Completed {len(WALK_FORWARD_FOLDS)} folds")

    # Save validation results as JSON
    validation_json_path = project_root / "storage" / "research" / "reports" / "walk_forward_results.json"
    save_validation_results(cv_results, WALK_FORWARD_FOLDS, validation_json_path)
    print(f"   Saved {validation_json_path}")

    # Train final models
    print(f"\n3. Training final models on GW 1-{FINAL_TRAIN_END}...")
    models = train_final_models(df, FINAL_TRAIN_END)
    for name, (model, meta) in models.items():
        print(f"   {name}: {meta['n_train']:,} training samples")

    # Save models
    print("\n4. Saving models...")
    models_dir = project_root / "storage" / "research" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, (model, meta) in models.items():
        model_path = models_dir / f"{name}_model.pkl"
        joblib.dump({"model": model, "metadata": meta}, model_path)
        print(f"   Saved {model_path}")

    # Generate beliefs
    print("\n5. Generating beliefs...")
    beliefs = predict_beliefs(df, models)

    # Quality checks
    print("\n6. Running quality checks...")
    targets = pd.read_csv(project_root / "storage" / "research" / "datasets" / "targets.csv")
    enforce_belief_contract(beliefs, targets)
    print("   All checks passed")

    # Save beliefs
    beliefs_path = project_root / "storage" / "research" / "datasets" / "beliefs.csv"
    beliefs.to_csv(beliefs_path, index=False)
    print(f"   Saved {beliefs_path}")

    # Generate report
    print("\n7. Generating evaluation report...")
    report_path = project_root / "storage" / "research" / "reports" / "stage5_belief_models.md"
    generate_report(cv_results, models, df, report_path)
    print(f"   Saved {report_path}")

    # Summary
    print("\n" + "=" * 50)
    print("Stage 5 Complete")
    print("=" * 50)
    print(f"\nOutputs:")
    print(f"  Models:     {models_dir}")
    print(f"  Beliefs:    {beliefs_path}")
    print(f"  Validation: {validation_json_path}")
    print(f"  Report:     {report_path}")

    # Quick stats
    print(f"\nBelief coverage:")
    for col in ["p_play", "p60", "mu_points", "p_haul"]:
        non_null = beliefs[col].notna().sum()
        print(f"  {col}: {non_null:,} / {len(beliefs):,} ({non_null/len(beliefs):.1%})")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for Stage 5."""
    project_root = Path(__file__).resolve().parents[4]
    build_belief_models(project_root)


if __name__ == "__main__":
    main()
