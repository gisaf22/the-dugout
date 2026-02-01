# The Dugout - Copilot Instructions

## Project Identity
**The Dugout** is decision support for Fantasy Premier League.
Three decisions: Captain, Transfer-In, Free Hit.
One rule: `argmax(predicted_points)`
Package name: `dugout`

## Supported Decisions
| Decision | Core Module | CLI | Rule |
|----------|-------------|-----|------|
| Captain | `src/dugout/production/decisions/captain.py` | `scripts/decisions/captain_cli.py` | Pick player with highest predicted_points |
| Transfer-In | `src/dugout/production/decisions/transfer.py` | `scripts/decisions/transfer_cli.py` | Rank players by predicted_points (exclude owned) |
| Free Hit | `src/dugout/production/decisions/free_hit.py` | `scripts/decisions/free_hit_cli.py` | LP maximize Σ(predicted_points) under budget/formation |

## Architecture Snapshot
- **Core Focus**: FPL decision support (Captain, Transfer-In, Free Hit)
- **Production pipeline**: `dugout.production` - Frozen, research-validated code
- **Research pipeline**: `dugout.research` - Experimental notebooks and analysis
- Data layer: `dugout.production.data` - API client, DB manager, data reader
- Features: `dugout.production.features` - Rolling features, feature builder
- Models: `dugout.production.models` - Predictor, captain, squad optimizer
- **Deferred**: Agentic/API components documented in `docs/DEFERRED_FEATURES.md`

## Core Components

### Data Layer (`src/dugout/production/data/`)
- `api_client.py` - HTTP client for FPL API with rate limiting
- `db_manager.py` - SQLite database schema and CRUD operations
- `puller.py` - Orchestrator facade (combines API + DB)
- `reader.py` - Read-only database access with caching
- `schemas.py` - Pydantic models for FPL data

### Features (`src/dugout/production/features/`)
- `builder.py` - Feature engineering for predictions
- `definitions.py` - Feature column definitions
- `team_form.py` - Dynamic team form calculations

### Models (`src/dugout/production/models/`)
- `captain.py` - Captain selection (argmax expected points)
- `squad.py` - Free Hit optimizer with PuLP linear programming
- `backtest.py` - Walk-forward validation
- `baseline.py` - Rolling average baseline model

### Analysis (`src/dugout/production/analysis/`)
- `models/` - Model-focused analysis (metrics.py, explainer.py)
- `decisions/` - Decision-focused analysis (decision_eval.py, regret_analysis.py)

## Data & Models
- Database: `storage/fpl_2025_26.sqlite` (current season); `DUGOUT_DB_PATH` env var overrides
- Production model: `storage/production/models/lightgbm_v1/model.joblib`
- `dugout.production.data.schemas` defines the canonical Pydantic models
- `DataReader` exposes read-only helpers with cached bulk history
- Player costs are stored in tenths (`now_cost`); FeatureBuilder converts to millions

## Decision Modules (`src/dugout/production/decisions/`)
Core logic implementing frozen rule: `argmax(predicted_points)`
- `captain.py` - `pick_captain()` function
- `transfer.py` - `recommend_transfers()` function
- `free_hit.py` - `optimize_free_hit()` function

## CLI Scripts (`scripts/`)

### Decision Scripts (`scripts/decisions/`)
- `captain_cli.py` - Weekly captain recommendation
- `transfer_cli.py` - Transfer-in recommendations  
- `free_hit_cli.py` - Optimize 15-player Free Hit squad

### Operations Scripts (`scripts/ops/`)
- `build_features.py` - Build training features from raw data
- `train_and_eval.py` - Train and evaluate production model
- `pull_fpl_data.py` - Fetch latest FPL data

### Backtest Scripts (`scripts/backtests/`)
- `models/walk_forward_validation.py` - Model accuracy over time
- `models/compare_models.py` - A/B model comparison
- `decisions/captain_backtest.py` - Captain regret evaluation
- `decisions/transfer_backtest.py` - Transfer regret evaluation
- `decisions/free_hit_backtest.py` - Free Hit regret evaluation

## Developer Workflow
- Install deps: `pip install -r requirements.txt`
- Ensure `PYTHONPATH` includes `src/` when running scripts
- Run tests: `pytest -q` (DB-dependent tests skip if sqlite missing)
- Train model: `PYTHONPATH=src python scripts/ops/train_and_eval.py`
- Captain pick: `PYTHONPATH=src python scripts/decisions/captain_cli.py --gw 24`
- Free hit: `PYTHONPATH=src python scripts/decisions/free_hit_cli.py --gw 24`

## Coding Conventions
- Keep comments minimal and focused on non-obvious reasoning
- Use Pydantic models with `model_validate`/`model_dump(exclude_none=True)`
- Maintain graceful fallbacks (e.g., catching forecast failures)
- Type hints encouraged throughout
