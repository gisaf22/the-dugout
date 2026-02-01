# The Dugout

Decision support for Fantasy Premier League Managers.

Pick the highest expected scorer. That's it.

## Supported Decisions

| Decision | Rule | CLI |
|----------|------|-----|
| **Captain** | Highest `predicted_points` | `scripts/decisions/captain_cli.py` |
| **Transfer-In** | Rank by `predicted_points` (exclude owned) | `scripts/decisions/transfer_cli.py` |
| **Free Hit** | LP-maximize Σ`predicted_points` under budget | `scripts/decisions/free_hit_cli.py` |

All decisions use the same frozen rule: `argmax(predicted_points)`.

## Non-Goals

- Ownership/differential strategies
- Multi-gameweek planning
- Chip timing optimization
- Price change predictions

## Quick Start

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/ops/pull_fpl_data.py   # Fetch FPL data
PYTHONPATH=src python scripts/decisions/captain_cli.py --gw 24
```

## Architecture

| | Production | Research |
|---|------------|----------|
| **Purpose** | Live predictions | Evidence generation |
| **Metrics** | MAE, RMSE | Regret (pts/GW) |

Production code is frozen. Research validates but does not deploy.

See [docs/production/GUARDRAILS.md](docs/production/GUARDRAILS.md) for model constraints.

## Project Structure

```
src/dugout/
├── production/           # Frozen prediction system
│   ├── data/             # API client, DB, reader
│   ├── features/         # Rolling features
│   ├── models/           # LightGBM predictor
│   ├── decisions/        # Captain, transfer, free hit
│   └── analysis/         # Metrics, regret analysis
└── research/             # Walk-forward validation
```

## Data

Not included. Run `scripts/ops/pull_fpl_data.py` to fetch from FPL API.

Requires `storage/fpl_2025_26.sqlite` with ≥5 completed gameweeks.

## License

MIT
