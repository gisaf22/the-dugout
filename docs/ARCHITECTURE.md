# The Dugout - Architecture

> Decision support for Fantasy Premier League.

## Core Concept

Three decisions. One rule: `argmax(predicted_points)`

| Decision | Description |
|----------|-------------|
| **Captain** | Pick player with highest expected points |
| **Transfer-In** | Rank players by expected points (exclude owned) |
| **Free Hit** | LP-optimize 15-player squad under budget/formation |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            THE DUGOUT                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐   ┌─────────────────┐   ┌────────────────────┐          │
│  │   DATA LAYER   │──▶│   PREDICTIONS   │──▶│     DECISIONS      │          │
│  └────────────────┘   └─────────────────┘   └────────────────────┘          │
│                                                                              │
│  ┌────────────────┐   ┌─────────────────┐   ┌────────────────────┐          │
│  │ fpl_2025_26.db │   │ predicted_pts   │   │ Captain            │          │
│  │ DataReader     │   │ p_play          │   │ Transfer-In        │          │
│  │ Pydantic       │   │ mu_points       │   │ Free Hit           │          │
│  │ schemas        │   │                 │   │                    │          │
│  └────────────────┘   └─────────────────┘   └────────────────────┘          │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  CORE DECISION MODULES                                                       │
│  └── src/dugout/production/decisions/                                        │
│      ├── captain.py    → pick_captain()                                      │
│      ├── transfer.py   → recommend_transfers()                               │
│      └── free_hit.py   → optimize_free_hit()                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  CLI INTERFACES                                                              │
│  └── scripts/decisions/                                                      │
│      ├── captain_cli.py                                                      │
│      ├── transfer_cli.py                                                     │
│      └── free_hit_cli.py                                                     │
│  API: Deferred (see DEFERRED_FEATURES.md)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Data Foundation

### Storage
- **Database**: `storage/fpl_2025_26.sqlite` - current season data
- **Environment**: `DUGOUT_DB_PATH` overrides database location

### Core Schemas (`src/dugout/production/data/schemas.py`)
```python
Player          # id, web_name, element_type, now_cost, status, team
GameweekEntry   # minutes, total_points, goals, assists, bps, ict_index
Fixture         # home_team, away_team, kickoff_time, difficulty
PlayerForecast  # predicted_points, expected_minutes, uncertainty
```

### Data Access (`src/dugout/production/data/reader.py`)
```python
DataReader.get_all_players()              # Current player pool
DataReader.get_players_recent_history_bulk()  # Last N GW history
DataReader.get_next_gameweek()            # Upcoming GW number
```

---

## Layer 2: ML Signal Generation

### Two-Stage Prediction Model
```
Stage 1: p_play    — P(minutes > 0)      — LightGBM classifier on all rows
Stage 2: mu_points — E[points | plays]   — LightGBM regressor on rows with minutes > 0

Final:  predicted_points = p_play × mu_points
```

This separates "did not play" from "played badly" — the research-validated approach.

### Signal Definitions

| Signal | Source | Description |
|--------|--------|-------------|
| `predicted_points` | Two-stage model | p_play × mu_points |
| `p_play` | Classifier | Probability of playing (minutes > 0) |
| `mu_points` | Regressor | Expected points conditional on playing |

### Feature Engineering (`src/dugout/production/features/`)
- Rolling statistics over last 5 games (mean, sum, variance)
- Recent form indicators (appearances, minutes fraction)
- Fixture context (home/away)

---

## Layer 3: Decision Aids

### 🎯 Captain Selection (`src/dugout/production/decisions/captain.py`)

Simple argmax-based selection:

Captain selection uses `argmax(predicted_points)` - the player with highest expected points is recommended.

```python
from dugout.production.decisions import pick_captain

captain = pick_captain(predictions_df, squad_ids)
```

### 🚀 Free Hit Optimizer (`src/dugout/production/models/squad.py`)

Pure EV maximization under FPL constraints:

```python
from dugout.production.models import FreeHitOptimizer

optimizer = FreeHitOptimizer(predictions_df, budget=100.0)
result = optimizer.optimize()
result.print_squad()
```

**Constraints enforced:**
- 15-player squad (2 GK, 5 DEF, 5 MID, 3 FWD)
- Valid formation (1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD)
- Max 3 players per team
- Budget ≤ £100m

---

## Pipeline Architecture

### Full Pipeline (`src/dugout/production/pipeline/`)

```
┌─────────────┐   ┌─────────────────┐   ┌─────────────┐   ┌─────────────┐
│  Extractor  │──▶│FeatureEngineer │──▶│   Trainer   │──▶│  Predictor  │
└─────────────┘   └─────────────────┘   └─────────────┘   └─────────────┘
     │                    │                    │                 │
     ▼                    ▼                    ▼                 ▼
  Raw data          Engineered           Trained           Predictions
  from DB            features             model              + signals
```

**Pipeline Modes:**
- `full`: Extract → Engineer → Train → Predict
- `train`: Load data → Engineer → Train
- `predict`: Load data → Predict (using existing model)
- `extract`: Extract data only

---

## File Structure

```
the-dugout/
├── src/dugout/
│   ├── production/            # Frozen production code
│   │   ├── data/              # API client, DB, data reader
│   │   ├── features/          # Feature engineering
│   │   ├── models/            # Squad optimizer, backtest runners
│   │   ├── decisions/         # Captain, transfer, free_hit logic
│   │   ├── analysis/          # Metrics & diagnostics
│   │   │   ├── models/        # MAE, RMSE, feature importance
│   │   │   └── decisions/     # Decision eval, regret analysis
│   │   └── pipeline/          # Training & evaluation
│   └── research/              # Research notebooks & validation
├── scripts/
│   ├── decisions/             # CLI wrappers
│   │   ├── captain_cli.py
│   │   ├── transfer_cli.py
│   │   └── free_hit_cli.py
│   ├── backtests/             # Backtest scripts
│   │   ├── models/            # Walk-forward, model comparison
│   │   └── decisions/         # Captain/transfer/free_hit regret
│   └── ops/                   # Data operations
│       ├── pull_fpl_data.py
│       ├── build_features.py
│       └── train_and_eval.py
├── storage/
│   ├── fpl_2025_26.sqlite     # Current season database
│   └── production/
│       ├── models/            # Trained models
│       └── reports/           # Evaluation outputs
└── tests/
    ├── production/            # Production tests
    │   ├── models/
    │   ├── decisions/
    │   ├── data/
    │   └── features/
    └── research/              # Research tests
```

---

## Signal Flow Example

```
User asks: "Who should I captain?"

1. DataReader loads player history (last 5 GWs)
2. FeatureBuilder computes rolling stats
3. Two-stage model predicts:
   - p_play = P(minutes > 0)
   - mu_points = E[points | plays]
   - predicted_points = p_play × mu_points
4. Decision: argmax(predicted_points)
```

---

## Key Design Decisions

### Why Two-Stage Prediction?
Minutes and points are fundamentally different:
- Participation: Rotation, fitness, manager preference (classification)
- Performance: Goals, assists, bonus (regression)

Separating them avoids the "Guardiola problem" where a benched player gets high expected points.

---

## Supported Decisions

| Decision | Status | Core Module | CLI |
|----------|--------|-------------|-----|
| Captain | ✅ Production | `src/dugout/production/decisions/captain.py` | `scripts/decisions/captain_cli.py` |
| Transfer-In | ✅ Production | `src/dugout/production/decisions/transfer.py` | `scripts/decisions/transfer_cli.py` |
| Free Hit | ✅ Production | `src/dugout/production/decisions/free_hit.py` | `scripts/decisions/free_hit_cli.py` |

All decisions use the frozen rule: `argmax(predicted_points)`

---

## Deferred Features

| Feature | Status | Notes |
|---------|--------|-------|
| Streamlit UI | ⏸️ Deferred | See DEFERRED_FEATURES.md |
| FastAPI service | ⏸️ Deferred | See DEFERRED_FEATURES.md |
| ADK agent | ⏸️ Deferred | See DEFERRED_FEATURES.md |

---

## Development Workflow

```bash
# Install dependencies
pip install -r requirements.txt

# Set database path (optional - defaults to storage/fpl_2025_26.sqlite)
export DUGOUT_DB_PATH=/path/to/custom.sqlite

# Run captain decision
PYTHONPATH=src python scripts/decisions/captain_cli.py --gw 24

# Run free hit optimization
PYTHONPATH=src python scripts/decisions/free_hit_cli.py --gw 24
```
