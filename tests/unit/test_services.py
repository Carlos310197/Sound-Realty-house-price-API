"""
Unit tests for service-layer functions.

All external dependencies (model, demographics, metadata) are mocked.
No file I/O, no HTTP, no real model loaded.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.api.v1.routes.health import check_health
from app.api.v1.routes.info import get_model_info
from app.api.v1.routes.predict import make_prediction


MOCK_METADATA = {
    "model_type": "KNeighborsRegressor",
    "version": "1.0.0",
    "features": [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
        "floors", "sqft_above", "sqft_basement",
        "ppltn_qty", "medn_hshld_incm_amt",
    ],
    "training_date": "2026-04-05",
    "rmse": 201659.43,
    "description": "KNN model",
}

MOCK_DEMOGRAPHICS = {
    "ppltn_qty": 38249.0,
    "medn_hshld_incm_amt": 66051.0,
}

HOUSE_FEATURES = {
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 1500.0,
    "sqft_lot": 4000.0,
    "floors": 1.0,
    "sqft_above": 1500.0,
    "sqft_basement": 0.0,
    "zipcode": "98042",
}


# ---------------------------------------------------------------------------
# check_health
# ---------------------------------------------------------------------------

class TestCheckHealth:

    def test_healthy_when_model_and_metadata_loaded(self):
        mock_model = MagicMock()
        mock_meta = MOCK_METADATA.copy()
        with patch("app.api.v1.routes.health.model_service.get_model", return_value=mock_model), \
             patch("app.api.v1.routes.health.model_service.get_metadata", return_value=mock_meta):
            result = check_health()
        assert result["status"] == "healthy"
        assert result["model_loaded"] is True

    def test_unhealthy_when_model_raises(self):
        with patch("app.api.v1.routes.health.model_service.get_model", side_effect=RuntimeError), \
             patch("app.api.v1.routes.health.model_service.get_metadata", side_effect=RuntimeError):
            result = check_health()
        assert result["status"] == "unhealthy"
        assert result["model_loaded"] is False

    def test_unhealthy_message_when_nothing_loaded(self):
        with patch("app.api.v1.routes.health.model_service.get_model", side_effect=RuntimeError), \
             patch("app.api.v1.routes.health.model_service.get_metadata", side_effect=RuntimeError):
            result = check_health()
        assert "model" in result["message"].lower()

    def test_returns_dict_with_required_keys(self):
        mock_model = MagicMock()
        with patch("app.api.v1.routes.health.model_service.get_model", return_value=mock_model), \
             patch("app.api.v1.routes.health.model_service.get_metadata", return_value=MOCK_METADATA):
            result = check_health()
        assert set(result.keys()) == {"status", "model_loaded", "message"}


# ---------------------------------------------------------------------------
# get_model_info
# ---------------------------------------------------------------------------

class TestGetModelInfo:

    def test_returns_metadata_dict(self):
        with patch("app.api.v1.routes.info.get_metadata", return_value=MOCK_METADATA):
            result = get_model_info()
        assert result == MOCK_METADATA

    def test_raises_value_error_when_metadata_unavailable(self):
        """RuntimeError from get_metadata must be re-raised as ValueError."""
        with patch("app.api.v1.routes.info.get_metadata", side_effect=RuntimeError("not loaded")):
            with pytest.raises(ValueError, match="not loaded"):
                get_model_info()


# ---------------------------------------------------------------------------
# make_prediction
# ---------------------------------------------------------------------------

class TestMakePrediction:

    def _mock_predict(self, return_value=450000.0):
        """Helper: returns context managers for a successful prediction."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [return_value]
        return (
            patch("app.api.v1.routes.predict.model_service.get_model", return_value=mock_model),
            patch("app.api.v1.routes.predict.model_service.get_metadata", return_value=MOCK_METADATA),
            patch("app.api.v1.routes.predict.demographics_service.get_demographics", return_value=MOCK_DEMOGRAPHICS),
        )

    def test_returns_float(self):
        p1, p2, p3 = self._mock_predict(450000.0)
        with p1, p2, p3:
            result = make_prediction(HOUSE_FEATURES.copy())
        assert isinstance(result, float)

    def test_prediction_is_rounded_to_two_decimals(self):
        p1, p2, p3 = self._mock_predict(450000.123456)
        with p1, p2, p3:
            result = make_prediction(HOUSE_FEATURES.copy())
        assert result == round(450000.123456, 2)

    def test_passes_zipcode_to_demographics(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [400000.0]
        mock_get_demo = MagicMock(return_value=MOCK_DEMOGRAPHICS)
        with patch("app.api.v1.routes.predict.model_service.get_model", return_value=mock_model), \
             patch("app.api.v1.routes.predict.model_service.get_metadata", return_value=MOCK_METADATA), \
             patch("app.api.v1.routes.predict.demographics_service.get_demographics", mock_get_demo):
            make_prediction(HOUSE_FEATURES.copy())
        mock_get_demo.assert_called_once_with("98042")

    def test_raises_value_error_when_model_not_loaded(self):
        with patch("app.api.v1.routes.predict.model_service.get_model", side_effect=RuntimeError("Model not loaded")):
            with pytest.raises(RuntimeError):
                make_prediction(HOUSE_FEATURES.copy())

    def test_raises_value_error_for_unknown_zipcode(self):
        mock_model = MagicMock()
        with patch("app.api.v1.routes.predict.model_service.get_model", return_value=mock_model), \
             patch("app.api.v1.routes.predict.model_service.get_metadata", return_value=MOCK_METADATA), \
             patch("app.api.v1.routes.predict.demographics_service.get_demographics",
                   side_effect=ValueError("Zipcode '00000' not found")):
            with pytest.raises(ValueError, match="not found"):
                make_prediction({**HOUSE_FEATURES, "zipcode": "00000"})
