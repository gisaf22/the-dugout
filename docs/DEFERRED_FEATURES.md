# Deferred Features - To Implement Later

> **Purpose**: Track agentic and API components removed during simplification for future re-implementation.  
> **Last Updated**: January 2026

---

## Overview

This project supports **three FPL decisions**:
- ✅ Captain selection
- ✅ Transfer-In recommendations
- ✅ Free Hit squad optimization

All decisions use the frozen rule: `argmax(predicted_points)`

The following **agentic/API components** were deferred for later implementation.

---

## 1. ADK Agent Integration (`adk/`)

### What It Did
- Google Agent Developer Kit (ADK) toolset for AI agent integration
- Exposed The Dugout capabilities as callable tools for LLM agents
- Allowed conversational FPL advice via agent orchestration

### Key Files (Removed)
```
adk/
├── agents/
│   └── fade_orchestrator/
│       ├── __init__.py
│       ├── agent.yaml          # Agent configuration
│       ├── fade_toolset.py     # Python toolset implementation
│       └── root_agent.yaml     # Root agent config
└── tools/
    ├── data_reader.yaml        # Tool manifest
    └── fade_toolset.py         # Shared toolset
```

### Key Implementation Details
- `FadeToolset` class extended `google.adk.tools.base_toolset.BaseToolset`
- Lazy initialization of shared `OrchestratorAgent`
- HTTP fallback when SQLite unavailable
- Tools exposed:
  - `players.list` - Paginated player listing
  - `players.get` - Single player detail
  - `players.recent` - Recent gameweek history
  - `fixtures.by_gw` - Fixtures for a gameweek
  - `forecast.next_gw` - Points predictions
  - `orchestrator.starting_eleven` - Lineup optimization

### Re-implementation Notes
- Requires `google-adk` package
- Set `DUGOUT_API_BASE` or `DUGOUT_DB_PATH` environment variables
- See [07-adk-integration-guide.md](./07-adk-integration-guide.md) for full guide

---

## 2. FastAPI Data Surface (`src/fade/api/`)

### What It Did
- RESTful API for accessing FPL data and predictions
- Endpoints for players, fixtures, forecasts, lineup optimization
- Dependency injection for DataReader/Orchestrator

### Key Files (Removed)
```
src/fade/api/
├── app.py          # FastAPI app with all endpoints
├── dependencies.py # DI factories for DataReader/Orchestrator
└── server.py       # Uvicorn server runner
```

### Endpoints (to restore)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/season/current` | Current season ID |
| GET | `/gameweek/next` | Next gameweek number |
| GET | `/players` | Paginated player list |
| GET | `/players/{id}` | Player detail |
| GET | `/players/{id}/recent` | Recent history |
| GET | `/players/{id}/last` | Last round entry |
| GET | `/fixtures` | Fixtures by gameweek |
| GET | `/forecast/next-gw` | Batch predictions |
| GET | `/orchestrator/starting-eleven` | Optimized lineup |

### Re-implementation Notes
- Requires `fastapi`, `uvicorn`, `httpx`
- Environment: `DUGOUT_DB_PATH`, `DUGOUT_API_HOST`, `DUGOUT_API_PORT`
- Start with: `uvicorn dugout.api.app:app --reload`

---

## 3. Orchestrator Agent (`src/fade/agents/`)

### What It Did
- High-level lineup optimization logic
- Formation enumeration (3-4-3, 4-3-3, etc.)
- Budget-constrained team selection
- Integration with forecasting for projected points

### Key Files (Removed)
```
src/fade/agents/
├── __init__.py
├── orchestrator.py       # FadeOrchestrator class
└── orchestrator_agent.py # OrchestratorAgent wrapper
```

### Key Classes
- `FadeOrchestrator` - Main lineup builder
  - `select_starting_eleven(budget, gameweek)` - Returns optimal 11
  - Uses formation options: 3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-3-2, 5-4-1
  - Position limits: 4 GK, 8 DEF, 8 MID, 6 FWD candidates
  
- `PlayerCandidate` - Dataclass with player + projected points
- `LineupPlayer` / `StartingEleven` - Pydantic response models

### Re-implementation Notes
- Similar logic exists in `FreeHitOptimizer` (using PuLP)
- Consider unifying lineup optimization approaches
- `orchestrator.py` lines 1-100 show the pattern

---

## 4. Heuristic Forecasting (`src/fade/forecasting.py`)

### What It Did
- Lightweight predictions without ML
- Rolling average blended with positional priors
- Fallback when ML model unavailable

### Status: **KEPT** ✅
This file remains as a baseline/fallback method.

---

## 5. Data Reader (`src/fade/tools/data_reader.py`)

### Status: **KEPT** ✅
Core data access layer remains. Used by all prediction components.

---

## Dependencies Removed

These can be removed from `requirements.txt` if not using agentic features:

```
# ADK (uncomment to restore)
# google-adk>=0.1.0

# API (uncomment to restore)
# fastapi>=0.100.0
# uvicorn>=0.23.0
# httpx>=0.24.0
```

---

## Restoration Checklist

When ready to add agentic features back:

1. [ ] Restore `adk/` folder from git history
2. [ ] Restore `src/fade/api/` folder from git history
3. [ ] Restore `src/fade/agents/` folder from git history
4. [ ] Add dependencies back to requirements.txt
5. [ ] Update `.github/copilot-instructions.md` with full context
6. [ ] Run tests: `pytest tests/test_api*.py tests/test_orchestrator*.py`

---

## Git Commands to Restore

```bash
# View what was removed
git log --oneline --all -- adk/ src/fade/api/ src/fade/agents/

# Restore specific folder from commit
git checkout <commit-hash> -- adk/
git checkout <commit-hash> -- src/fade/api/
git checkout <commit-hash> -- src/fade/agents/
```
