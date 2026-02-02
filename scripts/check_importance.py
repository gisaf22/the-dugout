#!/usr/bin/env python3
"""Check feature importance for production models."""

import joblib
from pathlib import Path

model_path = Path("storage/production/models/lightgbm_v2/two_stage_model.joblib")
model = joblib.load(model_path)

mu_model = model["mu_points"]
p_model = model["p_play"]

print("=== MU_POINTS MODEL (expected points) ===")
print("Feature Importance (gain):")
importance = dict(zip(mu_model.feature_name(), mu_model.feature_importance("gain")))
for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
    bar = "█" * int(imp / max(importance.values()) * 30)
    print(f"  {feat:22} {imp:8.0f} {bar}")

print()
print("=== P_PLAY MODEL (probability of playing) ===")
print("Feature Importance (gain):")
importance = dict(zip(p_model.feature_name(), p_model.feature_importance("gain")))
for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
    bar = "█" * int(imp / max(importance.values()) * 30)
    print(f"  {feat:22} {imp:8.0f} {bar}")
