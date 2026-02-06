"""Model registry for decision-specific models.

Provides get_model(decision) to load the appropriate model for each decision.
No fallbacks. No auto-selection. Explicit model per decision.

GUARDRAIL: Each decision loads a different model artifact.
           No shared prediction code paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

from dugout.production.config import MODEL_DIR
from dugout.production.models.base import BaseModel
from dugout.production.models.captain_model import CaptainModel
from dugout.production.models.transfer_model import TransferModel
from dugout.production.models.free_hit_model import FreeHitModel


# Type alias for decisions
Decision = Literal["captain", "transfer", "free_hit"]

# Default model directory
DEFAULT_MODEL_DIR = MODEL_DIR / "lightgbm_v2"


def get_model(decision: Decision, model_dir: Path = None) -> BaseModel:
    """Get the model for a specific decision.
    
    Each decision has its own model artifact. No fallbacks.
    
    Args:
        decision: One of "captain", "transfer", "free_hit"
        model_dir: Directory containing model artifacts (default: production)
        
    Returns:
        Decision-specific model instance
        
    Raises:
        ValueError: If decision is unknown
        FileNotFoundError: If model artifact doesn't exist
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    
    model_classes = {
        "captain": CaptainModel,
        "transfer": TransferModel,
        "free_hit": FreeHitModel,
    }
    
    if decision not in model_classes:
        raise ValueError(
            f"Unknown decision: {decision}. "
            f"Must be one of: {list(model_classes.keys())}"
        )
    
    model_class = model_classes[decision]
    return model_class.load(model_dir)


def list_available_models(model_dir: Path = None) -> dict:
    """List which decision models are available.
    
    Args:
        model_dir: Directory to check for models
        
    Returns:
        Dict mapping decision name to availability status
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    
    from dugout.production.models.captain_model import MODEL_FILENAME as CAPTAIN_FILE
    from dugout.production.models.transfer_model import MODEL_FILENAME as TRANSFER_FILE
    from dugout.production.models.free_hit_model import MODEL_FILENAME as FREE_HIT_FILE
    
    return {
        "captain": (model_dir / CAPTAIN_FILE).exists(),
        "transfer": (model_dir / TRANSFER_FILE).exists(),
        "free_hit": (model_dir / FREE_HIT_FILE).exists(),
    }


__all__ = ["get_model", "list_available_models", "Decision"]
