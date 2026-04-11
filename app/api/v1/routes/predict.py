from typing import Dict, Any
import pandas as pd
from app.services import model as model_service
from app.services import demographics as demographics_service


def make_prediction(house_features: Dict[str, Any]) -> float:
    """
    Make a price prediction for a single house.

    Args:
        house_features: Dictionary with all 18 columns from future_unseen_examples.csv.
                        Only the model's required features are used; the rest are ignored.

    Returns:
        Predicted price as a float.
    """
    if model_service.get_model() is None:
        raise ValueError("Model not loaded")

    # Enrich with demographics using zipcode, then drop zipcode
    zipcode = house_features["zipcode"]
    demographic_features = demographics_service.get_demographics(zipcode)

    all_features = {**house_features, **demographic_features}
    del all_features["zipcode"]

    # Extract feature values in the exact order used during training
    features = model_service.get_metadata()["features"]
    feature_values = {f: all_features[f] for f in features}

    X = pd.DataFrame([feature_values])

    prediction = model_service.get_model().predict(X)[0]

    return round(float(prediction), 2)
