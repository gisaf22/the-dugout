# The Dugout — Task Tracker

> Last Updated: January 29, 2026

---

## Current State: Production Ready

Three decisions supported, all using frozen `argmax(predicted_points)` rule.

### Decisions Status
| Decision | Description | Status |
|----------|-------------|--------|
| Captain | Weekly captain pick | ✅ Production |
| Transfer-In | Player recommendations | ✅ Production |
| Free Hit | Full squad optimization | ✅ Production |

---

## Recent Completed (Jan 2026)

| Task | Status | Notes |
|------|--------|-------|
| Decision-specific models | ✅ Done | Captain (18), Transfer (16), FreeHit (17) features |
| Epistemic alignment | ✅ Done | Research → Production |
| Model registry | ✅ Done | `get_model("captain")` |
| Captain backtest | ✅ Done | 4 pts/GW regret reduction |
| Free Hit optimizer | ✅ Done | LP with PuLP |
| Decision contract enforcement | ✅ Done | Runtime assertions |

---

## Backlog (Deferred)

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| D-01 | ADK agent integration | Low | See DEFERRED_FEATURES.md |
| D-02 | FastAPI service | Low | Deferred |
| D-03 | Streamlit UI | Low | Deferred |
| D-04 | Understat xG scraper | Medium | External data |
| D-05 | Multi-season training | Medium | ~250k rows |
| D-06 | Docker deployment | Medium | Production infra |

---

## Completed Archive

| Task | Completed | Result |
|------|-----------|--------|
| LightGBM model v1 | Dec 2025 | MAE 1.78 |
| 23 features | Jan 2026 | Rolling + interactions |
| OOP pipeline | Dec 2025 | 6 modules |
| Backtesting framework | Jan 2026 | Walk-forward |
| Regret analysis | Jan 2026 | Bucket diagnosis |
| Research pipeline | Jan 2026 | Stages 1-9 |
| Production pipeline | Jan 2026 | Frozen decisions |

---

## Notes

- Production code is frozen — changes require research validation
- Decision-specific models: CaptainModel, TransferModel, FreeHitModel
- Deferred features tracked in `DEFERRED_FEATURES.md`
