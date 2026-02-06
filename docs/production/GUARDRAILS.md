# Production Pipeline

## Model Configuration

### Decision-Specific Model Architecture

Each decision uses its own specialized model optimized for that use case:

| Model | Features | Use Case |
|-------|----------|----------|
| `CaptainModel` | 18 (position-conditional) | Captain pick |
| `TransferModel` | 16 (baseline) | Transfer recommendations |
| `FreeHitModel` | 17 (baseline + cost) | Free Hit optimization |

**Model location**: `storage/production/models/lightgbm_v2/`

### Model Files
| File | Description |
|------|-------------|
| `captain_model.joblib` | 18-feature position-conditional model |
| `transfer_model.joblib` | 16-feature baseline model |
| `free_hit_model.joblib` | 17-feature model with cost awareness |
| `model.joblib` | Legacy (kept for backward compatibility) |

### Model Registry
```python
from dugout.production.models import get_model

CaptainModel = get_model("captain")
TransferModel = get_model("transfer")
FreeHitModel = get_model("free_hit")
```

## Guardrails

### Decision-Model Binding
Each decision module loads its own model via registry:
- `captain.py` → `CaptainModel` via `get_model("captain")`
- `transfer.py` → `TransferModel` via `get_model("transfer")`
- `free_hit.py` → `FreeHitModel` via `get_model("free_hit")`

### Runtime Contract Assertions
Decision functions enforce:
- ❌ No research module imports
- ❌ No fixture difficulty signals
- ❌ No forbidden signals (p_play, p60, fixture_weight, weighted_ev)

Violations raise `RuntimeError` with explicit message.

### Model Logging
All decision outputs include:
- Model type in CLI output
- `Using data through GW{N}` context

## Decision Modules

| Module | Function | Rule |
|--------|----------|------|
| `src/dugout/production/decisions/captain.py` | `pick_captain()` | argmax(predicted_points) |
| `src/dugout/production/decisions/transfer.py` | `recommend_transfers()` | argmax(predicted_points) |
| `src/dugout/production/decisions/free_hit.py` | `optimize_free_hit()` | LP maximize Σ(predicted_points) |

## CLI Scripts

| Script | Model Used |
|--------|------------|
| `scripts/decisions/captain_cli.py` | CaptainModel |
| `scripts/decisions/transfer_cli.py` | TransferModel |
| `scripts/decisions/free_hit_cli.py` | FreeHitModel |

## Backtest Scripts

```bash
# Walk-forward validation
PYTHONPATH=src python scripts/backtests/models/walk_forward_validation.py

# Captain regret analysis
PYTHONPATH=src python scripts/backtests/decisions/captain_backtest.py
```

Results show per-GW and summary regret metrics.
