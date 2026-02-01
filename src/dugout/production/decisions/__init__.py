"""Production decision functions.

These are the frozen, research-validated decision rules.
Both CLI scripts and Streamlit app call these functions.

Decision Contract (Frozen):
- Captain: argmax(predicted_points)
- Transfer-IN: argmax(predicted_points)
- Free Hit: LP maximize Σ(predicted_points)

GUARDRAIL: All functions must use predict_points() from models.predict.
           Direct model loading is FORBIDDEN.
"""

from .captain import pick_captain, get_captain_candidates
from .transfer import get_transfer_recommendations
from .free_hit import optimize_free_hit

__all__ = [
    "pick_captain",
    "get_captain_candidates",
    "get_transfer_recommendations",
    "optimize_free_hit",
]
