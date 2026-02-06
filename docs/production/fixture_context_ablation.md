# Fixture Context Ablation (Production)

**Date**: 2025-01-20  
**Decision**: ❌ REJECT

## Summary

This ablation tests whether adding opponent-based fixture context features improves production decision quality.

**Features Added (2 total)**:
- `opp_xgc_per90_5`: Opponent's rolling 5-GW xG conceded (attacking context)
- `opp_xg_per90_5`: Opponent's rolling 5-GW xG (defensive context)

## Results

| Decision | Metric | Baseline | Fixture v1 | Delta | Verdict |
|----------|--------|----------|------------|-------|---------|
| Captain | Mean Regret | 15.38 pts/GW | 19.54 pts/GW | +4.16 | ❌ Worse |
| Captain | Hit Rate | 15.4% | 7.7% | -7.7pp | ❌ Worse |
| Transfer | Mean Regret | 9.00 pts/GW | 9.67 pts/GW | +0.67 | ❌ Worse |
| Transfer | Hit Rate | 5.6% | 5.6% | 0.0pp | — Same |
| Free Hit | Capture Rate | 48.6% | 47.2% | -1.4pp | ❌ Worse |
| Free Hit | Mean Regret | 69.00 pts/GW | 70.89 pts/GW | +1.89 | ❌ Worse |

## Key Observations

### Captain Performance Degraded Significantly
- Mean regret increased by **27%** (15.38 → 19.54)
- Hit rate halved (15.4% → 7.7%)
- Model now picks non-premium players (Rúben, Welbeck, Bijol instead of Haaland/Salah)

### Transfer Performance Slightly Degraded
- Marginal regret increase (+0.67 pts/GW)
- Hit rate unchanged at 5.6%

### Free Hit Marginally Degraded
- Capture rate dropped from 48.6% to 47.2%
- Mean regret increased slightly (+1.89 pts/GW)

## Hypothesis: Why Fixture Context Hurt

1. **Opponent stats are noisy at 5-GW window**: Small sample size makes rolling xG/xGC unreliable
2. **Fixture context may overfit to opponent patterns**: Model starts chasing "good fixture" players over consistent performers
3. **Captain picks shifted from premium to mid-tier**: The model may be up-weighting fixture context over proven attacking returns

## Decision

**REJECT** - Do not merge fixture context features into production.

All three decision types showed degradation. The features add noise rather than signal with the current implementation.

## Files Changed (to revert)

1. `src/dugout/production/features/definitions.py` - Remove `opp_xgc_per90_5`, `opp_xg_per90_5` from BASE_FEATURES
2. `src/dugout/production/features/builder.py` - Remove fixture context methods and FIXTURE_WINDOW constant

## Future Research Directions

If revisiting fixture context:
1. Try longer rolling windows (8-10 GWs)
2. Test separate models for attacking vs defensive players
3. Consider match-level features (home/away + opponent quality combined)
4. Validate on research pipeline before production integration

## Appendix: Per-GW Captain Comparison

### Baseline (15.38 regret)
| GW | Captain | Regret |
|----|---------|--------|
| 11 | Haaland | 18 |
| 12 | Haaland | 8 |
| 13 | Solanke | 26 |
| ... | ... | ... |

### Fixture v1 (19.54 regret)
| GW | Captain | Actual | Optimal | Regret |
|----|---------|--------|---------|--------|
| 11 | Haaland | 4 | Keane 15 | 22 |
| 12 | Welbeck | 8 | Muñoz 14 | 12 |
| 13 | Rúben | 1 | Foden 15 | 28 |
| 14 | Raúl | 2 | Tarkowski 10 | 16 |
| 15 | O'Reilly | 6 | B.Fernandes 18 | 24 |
| 16 | B.Fernandes | 13 | B.Fernandes 13 | 0 ✓ |
| 17 | Foden | 3 | Haaland 16 | 26 |
| 18 | Haaland | 2 | Gravenberch 11 | 18 |
| 19 | Wirtz | 3 | Matheus N. 11 | 16 |
| 20 | Rúben | 5 | Thiaw 17 | 24 |
| 21 | E.Le Fée | 0 | Wilson 9 | 18 |
| 22 | Bijol | 0 | Dorgu 15 | 30 |
| 23 | Bruno G. | 0 | Wilson 10 | 20 |
