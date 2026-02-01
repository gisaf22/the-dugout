"""
Research Report Builder — Evidence Collection

Collects final metrics from frozen pipeline outputs and compiles them into
a single JSON report for reproducibility.

This is read-only evidence extraction — no modeling or computation.

Output:
    storage/reports/research_report.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def load_walk_forward_results(project_root: Path) -> dict[str, Any]:
    """Load walk-forward validation results from Stage 5."""
    path = project_root / "storage" / "research" / "reports" / "walk_forward_results.json"
    if not path.exists():
        return {}
    
    with open(path) as f:
        data = json.load(f)
    
    # Extract average metrics per model
    metrics = {}
    for model_name, model_data in data.get("models", {}).items():
        model_results = model_data.get("model", [])
        if not model_results:
            continue
        
        if model_name in ("p_play", "p60", "p_haul"):
            # Classification: log_loss, brier_score
            avg_ll = sum(r["log_loss"] for r in model_results) / len(model_results)
            avg_brier = sum(r["brier_score"] for r in model_results) / len(model_results)
            metrics[model_name] = {
                "log_loss": round(avg_ll, 4),
                "brier_score": round(avg_brier, 4),
            }
        elif model_name == "mu_points":
            # Regression: MAE
            avg_mae = sum(r["mae"] for r in model_results) / len(model_results)
            metrics[model_name] = {
                "mae": round(avg_mae, 4),
            }
    
    return metrics


def load_captain_metrics(project_root: Path) -> dict[str, Any]:
    """Load captain evaluation metrics from Stage 6a."""
    path = project_root / "storage" / "research" / "datasets" / "evaluation_captain.csv"
    if not path.exists():
        return {}
    
    df = pd.read_csv(path)
    regret = df["regret"]
    
    return {
        "n_gameweeks": len(regret),
        "mean_regret": round(regret.mean(), 2),
        "median_regret": round(regret.median(), 2),
        "total_regret": int(regret.sum()),
        "zero_regret_rate": round((regret == 0).mean() * 100, 1),
        "high_regret_rate_gte_10": round((regret >= 10).mean() * 100, 1),
    }


def load_transfer_in_metrics(project_root: Path) -> dict[str, Any]:
    """Load transfer-IN evaluation metrics from Stage 7a."""
    path = project_root / "storage" / "research" / "datasets" / "evaluation_transfer_in.csv"
    if not path.exists():
        return {}
    
    df = pd.read_csv(path)
    
    # Group by policy
    metrics = {}
    for policy_name, group in df.groupby("policy_name"):
        regret = group["regret"]
        metrics[policy_name] = {
            "n_gameweeks": len(regret),
            "mean_regret": round(regret.mean(), 2),
            "zero_regret_rate": round((regret == 0).mean() * 100, 1),
        }
    
    return metrics


def load_multigw_hold_metrics(project_root: Path) -> dict[str, Any]:
    """Load multi-GW hold evaluation metrics from Stage 8b."""
    path = project_root / "storage" / "research" / "datasets" / "evaluation_multigw_hold.csv"
    if not path.exists():
        return {}
    
    df = pd.read_csv(path)
    
    # Group by horizon and policy
    metrics = {}
    for horizon in sorted(df["horizon"].unique()):
        horizon_key = f"H={horizon}"
        metrics[horizon_key] = {}
        
        horizon_df = df[df["horizon"] == horizon]
        for policy, group in horizon_df.groupby("policy"):
            regret = group["regret"]
            metrics[horizon_key][policy] = {
                "n_windows": len(regret),
                "mean_regret": round(regret.mean(), 2),
                "total_regret": int(regret.sum()),
            }
    
    return metrics


def build_report(project_root: Path) -> Path:
    """
    Build the consolidated research report.
    
    Collects metrics from all frozen outputs and writes to JSON.
    
    Args:
        project_root: Project root directory
    
    Returns:
        Path to the generated report
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "season": "2025/26",
        "belief_metrics": load_walk_forward_results(project_root),
        "captain": load_captain_metrics(project_root),
        "transfer_in": load_transfer_in_metrics(project_root),
        "multi_gw_hold": load_multigw_hold_metrics(project_root),
        "rejected_hypotheses": [
            "availability_weighted_ev_single_gw_captain",
            "fixture_context_adjustment",
            "p60_threshold_filtering",
            "safe_captain_p_start_threshold",
            "differential_captain_low_ownership",
        ],
        "accepted_policies": [
            "captain: argmax(mu_points)",
            "transfer_in: argmax(p_play × mu_points)",
            "multi_gw_hold: argmax(cum_ev) for H>=3",
        ],
    }
    
    output_path = project_root / "storage" / "research" / "reports" / "research_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return output_path


if __name__ == "__main__":
    from pathlib import Path
    
    # Resolve project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    
    report_path = build_report(project_root)
    print(f"Report written to: {report_path}")
    
    # Print summary
    with open(report_path) as f:
        report = json.load(f)
    
    print()
    print(json.dumps(report, indent=2))
