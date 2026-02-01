"""
Tests for Stage 5 — Belief Models

These tests verify:
1. Beliefs CSV contract (schema, row count, valid ranges)
2. p60 <= p_play constraint
3. mu_points/p_haul missingness matches expectation
4. Model artifacts exist and are loadable
5. Walk-forward validation metrics are reasonable
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import joblib

# Paths
BELIEFS_PATH = Path("storage/research/datasets/beliefs.csv")
TARGETS_PATH = Path("storage/research/datasets/targets.csv")
MODELS_DIR = Path("storage/research/models")

MODEL_NAMES = ["p_play_model.pkl", "p60_model.pkl", "mu_points_model.pkl", "p_haul_model.pkl"]


@pytest.fixture
def beliefs_df() -> pd.DataFrame:
    """Load beliefs CSV."""
    if not BELIEFS_PATH.exists():
        pytest.skip("beliefs.csv not found - run Stage 5 first")
    return pd.read_csv(BELIEFS_PATH)


@pytest.fixture
def targets_df() -> pd.DataFrame:
    """Load targets CSV for comparison."""
    if not TARGETS_PATH.exists():
        pytest.skip("targets.csv not found - run Stage 2 first")
    return pd.read_csv(TARGETS_PATH)


# =============================================================================
# Schema and Structure Tests
# =============================================================================


def test_beliefs_schema(beliefs_df: pd.DataFrame):
    """Beliefs CSV has correct columns."""
    expected = ["player_id", "gw", "p_play", "p60", "mu_points", "p_haul"]
    assert list(beliefs_df.columns) == expected, f"Expected {expected}, got {list(beliefs_df.columns)}"


def test_beliefs_row_count(beliefs_df: pd.DataFrame, targets_df: pd.DataFrame):
    """Beliefs has same row count as targets (one belief per prediction row)."""
    assert len(beliefs_df) == len(targets_df), (
        f"Row count mismatch: beliefs={len(beliefs_df)}, targets={len(targets_df)}"
    )


def test_beliefs_unique_key(beliefs_df: pd.DataFrame):
    """(player_id, gw) is unique in beliefs."""
    duplicates = beliefs_df.duplicated(subset=["player_id", "gw"], keep=False).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (player_id, gw) rows"


def test_beliefs_join_aligns(beliefs_df: pd.DataFrame, targets_df: pd.DataFrame):
    """Beliefs and targets join 1:1 on (player_id, gw)."""
    merged = beliefs_df.merge(
        targets_df[["player_id", "gw"]],
        on=["player_id", "gw"],
        how="outer",
        indicator=True
    )
    left_only = (merged["_merge"] == "left_only").sum()
    right_only = (merged["_merge"] == "right_only").sum()
    assert left_only == 0, f"Found {left_only} rows in beliefs not in targets"
    assert right_only == 0, f"Found {right_only} rows in targets not in beliefs"


# =============================================================================
# Probability Range Tests
# =============================================================================


def test_p_play_range(beliefs_df: pd.DataFrame):
    """p_play is in [0, 1]."""
    assert beliefs_df["p_play"].min() >= 0, "p_play has values < 0"
    assert beliefs_df["p_play"].max() <= 1, "p_play has values > 1"
    assert beliefs_df["p_play"].notna().all(), "p_play has NaN values"


def test_p60_range(beliefs_df: pd.DataFrame):
    """p60 is in [0, 1]."""
    assert beliefs_df["p60"].min() >= 0, "p60 has values < 0"
    assert beliefs_df["p60"].max() <= 1, "p60 has values > 1"
    assert beliefs_df["p60"].notna().all(), "p60 has NaN values"


def test_p_haul_range(beliefs_df: pd.DataFrame):
    """p_haul is in [0, 1] (where not NaN)."""
    valid = beliefs_df["p_haul"].dropna()
    if len(valid) > 0:
        assert valid.min() >= 0, "p_haul has values < 0"
        assert valid.max() <= 1, "p_haul has values > 1"


# =============================================================================
# Logical Constraint Tests
# =============================================================================


def test_p60_leq_p_play(beliefs_df: pd.DataFrame):
    """p60 <= p_play (can't play 60+ mins without playing at all)."""
    violations = beliefs_df[beliefs_df["p60"] > beliefs_df["p_play"]]
    assert len(violations) == 0, (
        f"Found {len(violations)} rows where p60 > p_play. "
        f"Max violation: {(violations['p60'] - violations['p_play']).max():.4f}"
    )


# =============================================================================
# Missingness Tests
# =============================================================================


def test_participation_beliefs_complete(beliefs_df: pd.DataFrame):
    """p_play and p60 have no missing values."""
    assert beliefs_df["p_play"].notna().all(), "p_play has missing values"
    assert beliefs_df["p60"].notna().all(), "p60 has missing values"


def test_performance_beliefs_missingness_aligned(beliefs_df: pd.DataFrame):
    """mu_points and p_haul are both NaN or both present."""
    mu_nan = beliefs_df["mu_points"].isna()
    haul_nan = beliefs_df["p_haul"].isna()
    misaligned = (mu_nan != haul_nan).sum()
    assert misaligned == 0, f"Found {misaligned} rows where mu_points/p_haul missingness differs"


def test_performance_beliefs_coverage(beliefs_df: pd.DataFrame, targets_df: pd.DataFrame):
    """Performance beliefs have reasonable coverage among those who played."""
    # Load targets to get y_play
    merged = beliefs_df.merge(targets_df[["player_id", "gw", "y_play"]], on=["player_id", "gw"])
    played = merged[merged["y_play"] == 1]
    
    if len(played) > 0:
        coverage = played["mu_points"].notna().mean()
        # Expect >90% coverage among those who played (some first appearances have no prior history)
        assert coverage > 0.90, f"mu_points coverage among played rows: {coverage:.1%}"


# =============================================================================
# mu_points Range Tests
# =============================================================================


def test_mu_points_reasonable_range(beliefs_df: pd.DataFrame):
    """mu_points is in reasonable range for FPL points (can be negative)."""
    valid = beliefs_df["mu_points"].dropna()
    if len(valid) > 0:
        # FPL points: typical range -2 to 20, rare extremes
        assert valid.min() > -5, f"mu_points min {valid.min():.2f} seems too low"
        assert valid.max() < 25, f"mu_points max {valid.max():.2f} seems too high"


def test_mu_points_not_all_same(beliefs_df: pd.DataFrame):
    """mu_points has variation (model isn't constant)."""
    valid = beliefs_df["mu_points"].dropna()
    if len(valid) > 100:
        assert valid.std() > 0.1, f"mu_points std {valid.std():.4f} is too low"


# =============================================================================
# Model Artifact Tests
# =============================================================================


def test_model_files_exist():
    """All model pickle files exist."""
    for name in MODEL_NAMES:
        path = MODELS_DIR / name
        assert path.exists(), f"Model file missing: {path}"


def test_models_loadable():
    """All model pickle files are loadable."""
    for name in MODEL_NAMES:
        path = MODELS_DIR / name
        if path.exists():
            try:
                obj = joblib.load(path)
            except Exception as e:
                pytest.fail(f"Failed to load {name}: {e}")
            
            # Check structure: should be dict with 'model' and 'metadata' keys
            assert isinstance(obj, dict), f"{name} is not a dict"
            assert "model" in obj, f"{name} missing 'model' key"
            model = obj["model"]
            
            # Basic sanity: model should have predict method
            assert hasattr(model, "predict"), f"{name} model has no predict method"


def test_models_have_metadata():
    """Model pickles include metadata dict."""
    for name in MODEL_NAMES:
        path = MODELS_DIR / name
        if path.exists():
            obj = joblib.load(path)
            assert isinstance(obj, dict), f"{name} is not a dict"
            assert "metadata" in obj, f"{name} missing 'metadata' key"
            meta = obj["metadata"]
            assert isinstance(meta, dict), f"{name} metadata is not a dict"
            assert "features" in meta, f"{name} missing 'features' in metadata"
            assert "target" in meta, f"{name} missing 'target' in metadata"


# =============================================================================
# Distribution Sanity Tests
# =============================================================================


def test_p_play_distribution(beliefs_df: pd.DataFrame, targets_df: pd.DataFrame):
    """p_play predictions are calibrated (mean ~ actual play rate)."""
    merged = beliefs_df.merge(targets_df[["player_id", "gw", "y_play"]], on=["player_id", "gw"])
    
    pred_mean = merged["p_play"].mean()
    actual_mean = merged["y_play"].mean()
    
    # Prediction mean should be within 10 percentage points of actual
    diff = abs(pred_mean - actual_mean)
    assert diff < 0.10, f"p_play mean {pred_mean:.3f} vs actual {actual_mean:.3f}"


def test_p60_distribution(beliefs_df: pd.DataFrame, targets_df: pd.DataFrame):
    """p60 predictions are calibrated (mean ~ actual 60+ min rate)."""
    merged = beliefs_df.merge(targets_df[["player_id", "gw", "y_60"]], on=["player_id", "gw"])
    
    pred_mean = merged["p60"].mean()
    actual_mean = merged["y_60"].mean()
    
    # Prediction mean should be within 10 percentage points of actual
    diff = abs(pred_mean - actual_mean)
    assert diff < 0.10, f"p60 mean {pred_mean:.3f} vs actual {actual_mean:.3f}"


def test_p_haul_low_base_rate(beliefs_df: pd.DataFrame):
    """p_haul predictions are low (hauls are rare ~5% of appearances)."""
    valid = beliefs_df["p_haul"].dropna()
    if len(valid) > 0:
        mean_haul = valid.mean()
        # Hauls should be rare; mean prediction should be < 20%
        assert mean_haul < 0.20, f"Mean p_haul {mean_haul:.3f} seems too high"


# =============================================================================
# Cross-Metric Sanity
# =============================================================================


def test_high_p_play_implies_higher_mu_points_coverage(beliefs_df: pd.DataFrame):
    """Players with high p_play should have mu_points more often (they have history)."""
    high_p_play = beliefs_df[beliefs_df["p_play"] > 0.8]
    low_p_play = beliefs_df[beliefs_df["p_play"] < 0.3]
    
    if len(high_p_play) > 100 and len(low_p_play) > 100:
        high_coverage = high_p_play["mu_points"].notna().mean()
        low_coverage = low_p_play["mu_points"].notna().mean()
        
        # High p_play players should have better mu_points coverage
        assert high_coverage >= low_coverage, (
            f"High p_play coverage ({high_coverage:.1%}) < low p_play coverage ({low_coverage:.1%})"
        )
