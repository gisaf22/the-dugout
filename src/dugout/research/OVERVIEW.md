# Research Pipeline

**Purpose**: Evidence generation and regret evaluation

## Overview

The research pipeline is the **evidence-driven, regret-evaluated system** that validates policies before they become production heuristics. It answers:

> "Does this belief or policy reduce regret?"

## Key Characteristics

| Aspect | Research |
|--------|----------|
| **Question** | What reduces regret? |
| **Metrics** | Regret (pts/GW) |
| **Mutability** | Frozen after validation |
| **Training** | Performance modeled conditional on participation |
| **Output** | `beliefs.csv`, `evaluation_*.csv` |

## The Fundamental Insight

The research pipeline exists because **modeling epistemology matters more than heuristics**.

The old production pipeline conflates "not playing" with "playing badly" by training on all rows with `per90 = 0` for non-appearances. The research pipeline explicitly separates:

1. **P(play)** — Will the player appear?
2. **E[points | plays]** — How many points if they do?

This separation, not a different decision rule, explains the large reduction in captain regret observed in Stage 6d.

## Pipeline Stages

```
research/pipeline/
├── targets.py                  # Stage 2: Ground-truth targets
├── features_participation.py   # Stage 3: Participation features
├── features_performance.py     # Stage 4a: Performance features
├── features_fixture_context.py # Stage 4b: Fixture context
├── belief_models.py            # Stage 5: Train belief estimators
├── belief_models_fixture.py    # Stage 5b: Fixture-aware beliefs
├── stage_6a_captain_policy.py  # Stage 6a: Captain evaluation
├── stage_6b_captain_baselines.py # Stage 6b: Baseline comparison
├── stage_6c_captain_revision.py  # Stage 6c: Policy revision
├── stage_7a_transfer_in.py     # Stage 7a: Transfer-in evaluation
├── stage_8a_multigw_beliefs.py # Stage 8a: Multi-GW beliefs
├── stage_8b_multigw_hold.py    # Stage 8b: Hold evaluation
├── run_all.py                  # Orchestrator
└── report_builder.py           # Evidence report generator
```

## Usage

```bash
# Run all stages end-to-end
PYTHONPATH=src python -m dugout.research.pipeline.run_all

# Run individual stage
PYTHONPATH=src python -m dugout.research.pipeline.belief_models
```

## Outputs

All outputs are frozen artifacts:

```
storage/research/
├── datasets/
│   ├── targets.csv                    # Stage 2
│   ├── features_participation.csv     # Stage 3
│   ├── features_performance.csv       # Stage 4a
│   ├── beliefs.csv                    # Stage 5
│   ├── evaluation_captain.csv         # Stage 6a
│   ├── evaluation_transfer_in.csv     # Stage 7a
│   └── evaluation_multigw_hold.csv    # Stage 8b
├── models/
│   ├── p_play_model.pkl
│   ├── p60_model.pkl
│   ├── mu_points_model.pkl
│   └── p_haul_model.pkl
└── reports/
    ├── research_report.json
    └── stage5_belief_models.md
```

## Key Evidence

From the research pipeline:

- **p_play log_loss**: 0.3239
- **mu_points MAE**: 2.21 pts/GW
- **Captain mean_regret**: 6.91 pts/GW
- **Low captain agreement** (11.8% with old pipeline) = strength, not weakness

### Rejected Hypotheses

The research pipeline explicitly rejects:
- `availability_weighted_ev_single_gw_captain` — EV-weighting hurts
- `fixture_context_adjustment` — No signal found
- `p60_threshold_filtering` — Reduces pool without improving regret

### Accepted Policies

- **Captain**: `argmax(mu_points)`
- **Transfer-in**: `argmax(mu_points)`
- **Multi-GW Hold (H=3–4 only)**: `argmax(Σ p_play × mu_points)`

## What This Is NOT

This pipeline does **not**:
- Power the Streamlit app
- Make real-time predictions
- Iterate on hyperparameters after validation

For app-facing predictions, see [`dugout.production`](../production/README.md).

## See Also

- [Decision Contracts](../../../docs/DECISION_CONTRACT_LAYER.md) — Frozen decision rules
- [Production Pipeline](../production/README.md) — Predictive system
