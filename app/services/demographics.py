import pandas as pd
from typing import Dict

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Module-level lookup table: zipcode (str) -> demographic feature dict
_demographics: Dict[str, Dict[str, float]] = {}


def load_demographics() -> None:
    """Load zipcode_demographics.csv into memory as a zipcode-keyed dict."""
    global _demographics
    df = pd.read_csv(settings.demographics_path, dtype={"zipcode": str})
    _demographics = df.set_index("zipcode").to_dict(orient="index")
    logger.info("Demographics data loaded successfully (%d zipcodes)", len(_demographics))


def get_demographics(zipcode: str) -> Dict[str, float]:
    """Return the demographic features for a given zipcode.

    Args:
        zipcode: 5-digit ZIP code string.

    Returns:
        Dict mapping each demographic column name to its value.

    Raises:
        ValueError: If the zipcode is not found in the demographics data.
    """
    if zipcode not in _demographics:
        raise ValueError(f"Zipcode '{zipcode}' not found in demographics data")
    return _demographics[zipcode]
