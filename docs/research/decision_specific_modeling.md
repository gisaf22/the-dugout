# Decision-Specific Modeling

> **Source of Truth** for prediction model variants per decision type.

## Executive Summary

Ablation testing (GW 11-23, n=13) demonstrates that **optimal prediction models differ by decision type**. This document captures the empirical evidence and implementation guidance.

**Key Insight**: Prediction models are decision-specific; decision rules are not.

The frozen rule `argmax(predicted_points)` applies uniformly across all decisions. What varies is *how* we compute `predicted_points` for each context.

---

## Ablation Results

### Full Walk-Forward Backtest (GW 11-23)

| Model | Captain Regret | Captain Hit% | Transfer Regret | Transfer Winner |
|-------|---------------|--------------|-----------------|-----------------|
| Baseline (16 feat) | 13.54 | 7.7% | **51.46** ✓ | ✓ |
| Defensive (18 feat) | 14.46 | **30.8%** | 54.23 | |
| **Conditional (18 feat)** | **11.69** ✓ | 7.7% | 54.69 | |

### Model Definitions

- **Baseline (16 feat)**: Production model, offensive per90 features only
- **Defensive (18 feat)**: Baseline + `xgc_per90` + `clean_sheet_rate` for all positions
- **Conditional (18 feat)**: Baseline + defensive features **only for DEF/GKP** (zeroed for MID/FWD)

---

## Why Captain ≠ Transfer

### Captain Context
- Selecting from **fixed 15-player squad** (includes DEF/GKP)
- Defensive features help rank DEF/GKP against attackers
- Position-conditional prevents noise in attacker rankings

### Transfer Context
- Ranking **entire player pool** (dominated by attackers)
- Defensive features add noise to MID/FWD comparisons
- Baseline's offensive focus aligns with transfer-in targets

### Why Hit Rate ≠ Regret

The defensive model achieves **4x higher hit rate** (30.8% vs 7.7%) but **higher regret** (14.46 vs 13.54).

This paradox occurs because:
1. Hit rate measures exact matches (binary)
2. Regret measures point differential (continuous)
3. A model that "almost hits" with low margins beats one that "hits more" but with large misses

**Regret is the authoritative metric** for decision quality.

---

## Recommendations

### Captain Decision
**Use position-conditional model**
- Add `xgc_per90`, `clean_sheet_rate` features
- Zero these features for MID/FWD (positions 3, 4)
- Improvement: **-1.85 pts/GW regret** vs baseline

### Transfer Decision
**Keep baseline model (16 features)**
- No defensive features
- No position conditioning
- Baseline is optimal for attacker-focused recommendations

### Free Hit Decision
**Keep baseline model + LP optimization**
- Squad optimization benefits from uniform prediction model
- LP handles position balancing, not feature engineering

---

## Implementation

### Architecture Overview

Each decision type loads a **separate model artifact** with **no shared prediction code paths**.

```
                    ┌─────────────────────┐
                    │  Decision Request   │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │   Captain    │   │   Transfer   │   │  Free Hit    │
    │  Decision    │   │  Decision    │   │  Decision    │
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ CaptainModel │   │TransferModel │   │FreeHitModel  │
    │  (18 feat)   │   │  (16 feat)   │   │  (17 feat)   │
    └──────────────┘   └──────────────┘   └──────────────┘
```

### Model Registry

```python
from dugout.production.models.registry import get_model

# Each call returns a decision-specific model instance
captain_model = get_model("captain")    # CaptainModel
transfer_model = get_model("transfer")  # TransferModel
free_hit_model = get_model("free_hit")  # FreeHitModel
```

### Feature Views

| Decision | Features | Key Difference |
|----------|----------|----------------|
| Captain | 18 (`CAPTAIN_FEATURES`) | + `xgc_per90`, `clean_sheet_rate` (position-conditional) |
| Transfer | 16 (`TRANSFER_FEATURES`) | Baseline only |
| Free Hit | 17 (`FREE_HIT_FEATURES`) | Baseline + `now_cost` |

### Position Conditioning (Captain Only)

```python
# In CaptainModel._apply_position_conditioning()
DEFENSIVE_POSITIONS = [1, 2]  # GKP, DEF

# xgc_per90 and clean_sheet_rate are zeroed for MID/FWD
# This prevents defensive noise in attacker rankings
```

### Model Artifacts

```
storage/production/models/lightgbm_v2/
├── captain_model.joblib    # 18 features, position-conditional
├── transfer_model.joblib   # 16 features, baseline
└── free_hit_model.joblib   # 17 features, includes cost
```

### Training

All training flows through canonical trainer:

```python
from dugout.production.pipeline.trainer import train_all_models

# Train and save all three decision-specific models
train_all_models(train_df)
```

### Decision Wiring

- `pick_captain()` → `get_model("captain")` → CaptainModel
- `get_transfer_recommendations()` → `get_model("transfer")` → TransferModel
- `optimize_free_hit()` → `get_model("free_hit")` → FreeHitModel

---

## Guardrails

### What Changes
- Separate model artifact per decision
- Each decision loads ONLY its own model
- No shared prediction code paths
- Position conditioning at inference time (captain only)

### What Does NOT Change
- Decision rule: `argmax(predicted_points)` (frozen)
- One change in one decision cannot affect another
- Training pipeline: single canonical trainer
- Feature builder: shared, decision-agnostic

### Explicit Non-Goals
- ❌ Model unification across decisions
- ❌ Fixture-based heuristics
- ❌ Availability weighting in predictions
- ❌ Hit rate optimization

---

## Validation Criteria

Before promoting captain_conditional to production:

1. **Captain regret ≤ baseline** on held-out GWs
2. **Transfer regret unchanged** (uses separate model)
3. **Free Hit regret unchanged** (uses baseline)
4. **No feature leakage** across decision boundaries

---

## References

- Ablation notebook: `notebooks/research/stage_4d_defensive_features_ablation.ipynb`
- Production guardrails: `docs/production/GUARDRAILS.md`
- Feature views: `src/dugout/production/features/views.py`
- Model registry: `src/dugout/production/models/registry.py`
- Decision-specific models:
  - Captain: `src/dugout/production/models/captain_model.py`
  - Transfer: `src/dugout/production/models/transfer_model.py`
  - Free Hit: `src/dugout/production/models/free_hit_model.py`
- Canonical trainer: `src/dugout/production/pipeline/trainer.py`
