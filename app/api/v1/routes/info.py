from typing import Dict, Any
from app.services.model import get_metadata

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.

    Returns:
        Dictionary containing model metadata

    This function should simply return the loaded metadata dictionary.
    If metadata is not loaded, it should raise an error.
    """
    try:
        return get_metadata()
    except RuntimeError as e:
        raise ValueError(str(e))