# Production Pipeline

## Model Configuration

### Current State
- **Default model**: `two_stage` (p_play × mu_points)
- **Fallback model**: `legacy` (single GBM)
- **Model location**: See `config.py:DEFAULT_MODEL_PATH`

### Model Files
| File | Description | Status |
|------|-------------|--------|
| `two_stage_model.joblib` | Epistemically-aligned model | Primary |
| `model.joblib` | Legacy single-stage model | Fallback |

### Selection Logic
```python
if two_stage_model.joblib exists:
    use two_stage (p_play × mu_points)
else:
    use legacy (single GBM)
```

## Guardrails

### Single Inference Entry Point
- All decisions MUST use `predict_points()` from `dugout.production.models.predict`
- Direct model loading is FORBIDDEN in decision scripts
- Backtests use their own training loop but respect `model_type` parameter

### Runtime Contract Assertions
Decision functions enforce:
- ❌ No research module imports
- ❌ No fixture difficulty signals
- ❌ No availability weighting
- ❌ No forbidden signals (p_play, p60, fixture_weight, weighted_ev)

Violations raise `RuntimeError` with explicit message.

### Model Type Logging
All decision outputs include:
- `model_type` field in CSV exports
- `Model: two_stage` in CLI output
- `Using data through GW{N}` context

## Kill Switch (Rollback)

To revert to legacy model:
```bash
cd storage/production/models/<current_version>
mv two_stage_model.joblib two_stage_model.joblib.disabled
```

Decision scripts automatically fall back to `model.joblib`.

## Removal Criteria

Two-stage model becomes sole model when:
1. ✅ Backtest regret improvement validated (4 pts/GW lower)
2. ⬜ 3+ live GWs without rollback needed
3. ⬜ No silent failures in production logs

Until all criteria met, both models remain available.

## Decision Modules

| Module | Function | Rule |
|--------|----------|------|
| `src/dugout/production/decisions/captain.py` | `pick_captain()` | argmax(predicted_points) |
| `src/dugout/production/decisions/transfer.py` | `recommend_transfers()` | argmax(predicted_points) |
| `src/dugout/production/decisions/free_hit.py` | `optimize_free_hit()` | LP maximize Σ(predicted_points) |

## CLI Scripts

| Script | Model Used | Outputs model_type |
|--------|------------|-------------------|
| `scripts/decisions/captain_cli.py` | auto-detect | ✅ |
| `scripts/decisions/free_hit_cli.py` | auto-detect | ✅ |
| `scripts/backtests/decisions/captain_backtest.py` | `--model-type` flag | ✅ |

## Backtest Comparison

```bash
# Compare legacy vs two-stage regret
PYTHONPATH=src python scripts/backtests/decisions/captain_backtest.py --compare-models
```

Results show per-GW and summary regret metrics for both models.
