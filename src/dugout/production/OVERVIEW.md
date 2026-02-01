# Production Pipeline

**Purpose**: Live prediction and app support

## Overview

The production pipeline is the **predictive, app-facing system** that powers The Dugout's real-time outputs. It answers:

> "What prediction should I output right now?"

This pipeline prioritizes **responsiveness, simplicity, and deployability** over epistemic precision. It is intentionally decoupled from the research pipeline, which exists to validate decision quality.

## Key Characteristics

| Aspect | Production |
|--------|------------|
| **Question** | What to predict right now? |
| **Metrics** | MAE, RMSE |
| **Mutability** | Iterative improvement |
| **Training** | Unconditional (all rows, played or not) |
| **Output** | `predicted_points` |

> ⚠️ **Important**: Production predictions conflate appearance risk and performance into a single estimate. This is acceptable for live outputs but is *not* epistemically equivalent to modeling performance conditional on participation.

## Components

```
production/
├── __init__.py
├── pipeline/
│   ├── __init__.py
│   ├── runner.py      # End-to-end orchestration
│   ├── trainer.py     # LightGBM training
│   └── evaluator.py   # MAE / RMSE evaluation
├── features/          # Production-oriented feature engineering
└── models/            # Trained prediction models
```

## Usage

```python
from dugout.production import Pipeline

# Run full pipeline
pipeline = Pipeline.run()

# Or step by step
pipeline = Pipeline()
pipeline.gather_data()
pipeline.build_features()
pipeline.split()
pipeline.train()
pipeline.evaluate()
pipeline.save_artifacts()
```

## Modeling Assumption (Explicit)

The production pipeline optimizes for predictive accuracy under the assumption that:

> Errors from non-appearance and errors from underperformance can be treated uniformly.

Research evidence shows this assumption is suboptimal for decision-making, but it remains acceptable for:
- Live ranking displays
- UI-driven captain suggestions
- Fast iteration cycles

**Validated decision rules should always come from the research pipeline.**

## What This Is NOT

This pipeline does **not**:
- Evaluate regret
- Compare policy alternatives
- Validate decision rules
- Separate participation from performance uncertainty

For evidence-based policy validation, see [`dugout.research`](../research/README.md).

## See Also

- [Decision Contracts](../../../docs/DECISION_CONTRACT_LAYER.md) — Frozen decision rules
- [Research Pipeline](../research/README.md) — Evidence-driven system
