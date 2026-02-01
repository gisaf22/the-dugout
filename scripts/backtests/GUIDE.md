# Backtests

Historical simulations where **time moves forward**. No future leakage.

This directory contains two fundamentally different types of backtests.
Understanding the distinction is critical for interpreting results correctly.

---

## What Is a Backtest?

A backtest is a **time-respecting historical simulation**:

1. Train/fit only on data available at time T
2. Make prediction/decision for time T+1
3. Observe outcome at T+1
4. Slide window forward, repeat

The key property: **no future information leaks into past decisions**.

This answers: *"Would this model/policy have worked in the past?"*

---

## What Is NOT a Backtest

| Activity | Why It's Not a Backtest |
|----------|-------------------------|
| Single-GW prediction | No time movement |
| Feature inspection | No simulation |
| Holdout evaluation without time ordering | Potential leakage |
| Offline analysis on shuffled data | Breaks temporal causality |

---

## Two Types of Backtests

### 1. Model Backtests (`model/`)

**Question answered:** *How accurate are the predictions over time?*

Model backtests evaluate **predictive stability** — whether the model's point
estimates track reality across unseen gameweeks.

- Uses walk-forward validation (train on past, predict future)
- Metrics: MAE, RMSE, Spearman correlation, calibration error
- No decisions are made; predictions are compared directly to outcomes

**What this tells you:**
- Is the model getting better or worse over time?
- Are predictions systematically biased?
- Is variance consistent across gameweeks?

**What this does NOT tell you:**
- Whether captain picks were good
- Whether transfers were optimal
- Whether Free Hit squads performed well

---

### 2. Decision Backtests (`decisions/`)

**Question answered:** *How good were the decisions given the predictions?*

Decision backtests evaluate **policy quality** — whether the frozen decision rule
produced good outcomes compared to a hindsight oracle.

- Uses the **same decision functions** as production (no duplication)
- Requires an **oracle** (hindsight-optimal choice under same constraints)
- Metrics: regret = oracle_points − chosen_points

**What is regret?**

```
Regret = Oracle Points − Chosen Points
```

The oracle is the optimal choice if we had perfect foresight. Low regret = good decisions.

**What this tells you:**
- How much value did we leave on the table?
- Did the decision rule pick the right players in hindsight?
- Is the policy stable or volatile?

**What this does NOT tell you:**
- Whether predictions were accurate (that's model backtest)
- Whether a different model would have helped

---

## Why Walk-Forward Is Still a Backtest

Some confusion arises because "backtest" sounds like looking backward, but
walk-forward moves forward through time. The term "backtest" refers to:

- **Back** = historical data (the past)
- **Test** = evaluate performance

Walk-forward validation is a backtest because it evaluates on historical data
while respecting temporal order. Time moves forward *within* the historical window.

---

## Why They Are Separated

A model can have low MAE but still produce bad decisions (if errors cluster
at decision-critical points). Conversely, a model with higher MAE might
still yield optimal decisions if its ranking is correct.

**Model accuracy ≠ Decision quality**

By separating these backtests, we can diagnose problems precisely:

| Symptom | Model Backtest | Decision Backtest | Likely Cause |
|---------|----------------|-------------------|--------------|
| High regret, low MAE | ✓ Passing | ✗ Failing | Decision rule issue |
| High regret, high MAE | ✗ Failing | ✗ Failing | Model issue |
| Low regret, high MAE | ✗ Failing | ✓ Passing | Ranking is correct despite point errors |

---

## Directory Structure

```
scripts/backtests/
├── README.md                         # This file
├── models/
│   ├── walk_forward_validation.py    # Prediction accuracy over time
│   └── compare_models.py             # A/B model comparison
└── decisions/
    ├── captain_backtest.py           # Captain selection regret
    ├── transfer_backtest.py          # Transfer-IN regret
    └── free_hit_backtest.py          # Free Hit squad regret
```

---

## When to Run Which

| Situation | Run This |
|-----------|----------|
| Before deployment | Decision backtests (all 3) |
| Before changing training | Model backtest |
| Before changing decision rules | Research pipeline (not production) |
| Weekly health check | Model backtest (quick) |
| Full credibility audit | All backtests |

---

## Commands

### Model Backtest

```bash
# Walk-forward validation (MAE, Spearman, calibration)
PYTHONPATH=src python scripts/backtests/models/walk_forward_validation.py

# A/B model comparison (single vs two-stage)
PYTHONPATH=src python scripts/backtests/models/compare_models.py
```

### Decision Backtests

```bash
# Captain backtest
PYTHONPATH=src python scripts/backtests/decisions/captain_backtest.py

# Transfer-IN backtest
PYTHONPATH=src python scripts/backtests/decisions/transfer_backtest.py
PYTHONPATH=src python scripts/backtests/decisions/transfer_backtest.py --start-gw 6 --end-gw 22

# Free Hit backtest
PYTHONPATH=src python scripts/backtests/decisions/free_hit_backtest.py
PYTHONPATH=src python scripts/backtests/decisions/free_hit_backtest.py --start-gw 6 --end-gw 22
```

---

## Decision Backtest Details

### Captain Backtest

| Property | Value |
|----------|-------|
| Decision Rule | `argmax(predicted_points)` |
| Oracle | Player with max actual points in squad |
| Metric | Regret (doubled points) |
| Output | Per-GW captain picks with regret |

### Transfer-IN Backtest

| Property | Value |
|----------|-------|
| Decision Rule | `argmax(predicted_points)` among eligible |
| Oracle | Player with max actual points among eligible |
| Eligibility | Available status (!= n, i, s, u) |
| Metric | Regret |
| Output | `storage/production/reports/evaluation_transfer_backtest.csv` |

### Free Hit Backtest

| Property | Value |
|----------|-------|
| Decision Rule | LP maximize Σ(predicted_points), mode=basic |
| Oracle | LP maximize Σ(actual_points), same constraints |
| Constraints | Budget (£100m), formation, team limits |
| Evaluation | Starting XI actual points only (bench ignored) |
| Metric | Regret, capture rate |
| Output | `storage/production/reports/evaluation_free_hit_backtest.csv` |

---

## What This Directory Is For

- Evaluating model and decision quality on historical data
- Diagnosing where the pipeline is underperforming
- Validating before deployment
- Building credibility through reproducible evaluation

## What This Directory Is NOT For

- Training models (see `scripts/ops/train_and_eval.py`)
- Making production decisions (see `scripts/decisions/`)
- Exploratory research (see `notebooks/`)
- Changing decision rules (requires research pipeline validation)
