import joblib
import json
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_model: Optional[Any] = None
_metadata: Optional[Dict[str, Any]] = None


def load_model_and_metadata() -> None:
    """Load the trained model and metadata from disk."""
    global _model, _metadata

    _model = joblib.load(settings.model_path)

    with open(settings.model_metadata_path, 'r') as f:
        _metadata = json.load(f)

    logger.info("Model and metadata loaded successfully!")


def get_model() -> Any:
    if _model is None:
        raise RuntimeError("Model not loaded — did startup run?")
    return _model


def get_metadata() -> Dict[str, Any]:
    if _metadata is None:
        raise RuntimeError("Metadata not loaded — did startup run?")
    return _metadata
