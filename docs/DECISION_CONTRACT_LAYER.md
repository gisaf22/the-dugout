# Decision Contract Layer

This document defines the frozen decision contracts that govern production behavior in The Dugout.

**Any change to a decision rule requires re-validation via the research pipeline.**

---

## 🔒 Production Contracts (Frozen)

**Last Updated**: 2026-01-23  
**Status**: FROZEN — Do not modify decision logic without re-running research pipeline

| Component | Rule | Rationale |
|-----------|------|-----------|
| **Captain** | `argmax(predicted_points)` | Stage 6 validated |
| **Transfer-IN** | `argmax(predicted_points)` | Stage 7a validated |
| **Free Hit** | LP maximize Σ(predicted_points) | Constrained optimization |
| **Availability** | Status filter only (`["n","i","s","u"]` excluded) | NOT weighted into EV |
| **Fixtures** | ❌ Explicitly NOT used | No signal found (Stage 4b) |
| **Research imports** | ❌ None allowed | Boundary enforced |
| **Inputs** | SQLite DB (`storage/fpl_*.sqlite`) | Single source of truth |
| **Outputs** | CSV files, in-memory DataFrames | No external dependencies |

### Decision Modules (Frozen)

```
src/dugout/production/decisions/
├── captain.py     # pick_captain() → argmax(predicted_points)
├── transfer.py    # recommend_transfers() → argmax(predicted_points)
└── free_hit.py    # optimize_free_hit() → LP maximize Σ(predicted_points)
```

**CLI Entry Points** (thin wrappers that call the above):
```
scripts/decisions/
├── captain_cli.py
├── transfer_cli.py
└── free_hit_cli.py
```

### What This Contract Prohibits

1. **Availability-weighted EV** (`p_play × points`) — rejected by research
2. **Fixture difficulty adjustments** — no predictive signal
3. **p60 threshold filtering** — reduces pool without reducing regret
4. **Importing `dugout.research.*`** — boundary violation

**This contract is the point of no return for deployment.**

---

## The Core Distinction

| Dimension | Production | Research |
|-----------|------------|----------|
| **Question** | What prediction should I output? | Does this policy reduce regret? |
| **Metric** | MAE / RMSE | Regret (pts/GW) |
| **Training data** | All rows | Conditional on event |
| **Outputs** | `predicted_points` | `beliefs.csv`, `evaluation_*.csv` |
| **Mutability** | Iterative | Frozen after validation |
| **Consumer** | Streamlit app | Jupyter notebooks, reports |

## Why the Split Exists

### The Problem

Before this refactor, a single `pipeline/` directory contained both:
- Production orchestration (`runner.py`, `trainer.py`)
- Research stages (`targets.py`, `belief_models.py`, `stage_6*.py`)

This created **conceptual debt**:
1. Unclear which code powered the app vs. generated evidence
2. Risk of production code importing unvalidated research logic
3. No visible boundary between "prediction" and "decision evaluation"

### The Insight

The research pipeline discovered that the **old production pipeline conflates two distinct phenomena**:

- Players who don't play (`minutes = 0`)
- Players who play but score poorly

By training on all rows with `per90 = 0` for non-appearances, the old model blends these cases. The research pipeline separates them:

```
P(play)      → Binary classifier on all rows
E[pts|play]  → Regressor on rows where y_play == 1
```

This **explicit separation of participation and conditional performance** drove the 43% regret reduction in captain selection.

## The Contract

### Production → Research: Never

Production code must **never** import from `dugout.research`. Research outputs are frozen artifacts, not live dependencies.

### Research → Production: Through Artifacts Only

When research validates a new policy, it graduates to production through:
1. Documented evidence (e.g., `docs/observations/*.md`)
2. Frozen model artifacts (e.g., `storage/models/*.pkl`)
3. Explicit code migration (not import)

### Shared Utilities

Non-epistemic utilities live in the stable layers:
- `dugout.data` — Database access
- `dugout.features` — Feature definitions
- `dugout.models` — Model utilities (not training)
- `dugout.config` — Paths and constants

These are safe to import from either pipeline.

## Directory Structure

```
src/dugout/
├── production/           # Predictive, app-facing
│   ├── pipeline/
│   │   ├── runner.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── README.md
│
├── research/             # Evidence, regret-evaluated
│   ├── pipeline/
│   │   ├── targets.py
│   │   ├── features_*.py
│   │   ├── belief_models.py
│   │   ├── stage_*.py
│   │   ├── run_all.py
│   │   └── report_builder.py
│   └── README.md
│
├── pipeline/             # DEPRECATED: Backward compatibility
│   └── __init__.py       # Re-exports from production
│
├── data/                 # Shared: Database access
├── features/             # Shared: Feature definitions
├── models/               # Shared: Model utilities
└── config.py             # Shared: Paths
```

## Backward Compatibility

The old `dugout.pipeline` module is preserved as a compatibility layer:

```python
# Old import (still works)
from dugout.pipeline import Pipeline

# New import (preferred)
from dugout.production import Pipeline
```

## Verification

To verify the boundary is maintained:

```bash
# Run research pipeline end-to-end
PYTHONPATH=src python -m dugout.research.pipeline.run_all

# Verify no research imports in production
grep -r "from dugout.research" src/dugout/production/
# Should return nothing
```

## The Epistemic Imperative

This refactor is not cosmetic. It enforces the insight that:

> **Modeling epistemology matters more than heuristics.**

The filesystem now reflects that truth.

## Research-Validated Production Defaults

The following production behaviors are explicitly validated by the research pipeline:

| Decision | Production Rule | Evidence | Notes |
|----------|-----------------|----------|-------|
| Captain (single GW) | `argmax(expected_points)` | Stage 6 | Do NOT weight by availability |
| Transfer-IN (single GW) | `argmax(expected_points)` | Stage 7a | Availability weighting increases regret |
| Availability weighting | ❌ Not used | Stage 6, 7 | Rejected for single-GW decisions |
| Fixture difficulty | ❌ Not used | Stage 4b | No signal found |
| p60 filtering | ❌ Not used | Diagnostic | Reduces pool without improving regret |
| Multi-GW holds | Not implemented | Stage 8 | Availability compounds over H=3–4 only |

### Why This Matters

These defaults are not arbitrary. Each was tested against alternatives:

- **Availability-weighted EV** (`p_play × mu_points`) was explicitly rejected for single-GW decisions
- **Fixture context adjustment** showed no predictive signal
- **p60 threshold filtering** reduced candidate pool without reducing regret

Future contributors should not reintroduce these heuristics without re-running the research pipeline.

**This document is authoritative.**

## Why Production Does Not Import Research Beliefs

- **Research beliefs are frozen artifacts, not live dependencies.** They exist to validate decisions, not to power the app.
- **Production prioritizes simplicity and deployability.** A single `predicted_points` column is easier to maintain than a multi-model belief stack.
- **The decision rule is what matters.** Research proved that `argmax(expected_points)` minimizes regret — production implements this rule, regardless of how `expected_points` is computed.
- **Conditional modeling adds complexity without app-level benefit.** Separating `p_play` and `mu_points` improves regret analysis but doesn't change the captain selection outcome.
- **Avoiding tight coupling preserves iteration speed.** Production can retrain its model without re-running the full research pipeline.

If a future version needs research-grade predictions in production, the path is:
1. Export frozen belief models as artifacts
2. Load them in production as read-only dependencies
3. Never import research code directly
