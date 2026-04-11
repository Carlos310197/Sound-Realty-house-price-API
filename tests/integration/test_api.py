"""
Integration tests for the House Price Prediction API.

These tests exercise the full request→service→response pipeline using
FastAPI's TestClient, which triggers the startup lifecycle (model + demographics
loading) and validates HTTP contracts end-to-end.
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Module-scoped client that triggers the startup event once, loading the
    real model and demographics data for all tests in this module.
    """
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "bedrooms": 4,
    "bathrooms": 1.0,
    "sqft_living": 1680,
    "sqft_lot": 5043,
    "floors": 1.5,
    "waterfront": 0,
    "view": 0,
    "condition": 4,
    "grade": 6,
    "sqft_above": 1680,
    "sqft_basement": 0,
    "yr_built": 1911,
    "yr_renovated": 0,
    "zipcode": "98042",  # exists in data/zipcode_demographics.csv
    "lat": 47.5354,
    "long": -122.273,
    "sqft_living15": 1560,
    "sqft_lot15": 5765,
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_schema(self, client):
        data = client.get("/health").json()
        assert set(data.keys()) == {"status", "model_loaded", "message"}

    def test_healthy_when_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert isinstance(data["message"], str)

    def test_unhealthy_when_model_not_loaded(self):
        """When get_model raises RuntimeError the service must report unhealthy, not crash."""
        with patch("app.services.model.get_model", side_effect=RuntimeError("Model not loaded")), \
             patch("app.services.model.get_metadata", side_effect=RuntimeError("Metadata not loaded")):
            with TestClient(app, raise_server_exceptions=False) as c:
                data = c.get("/health").json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# /model/info
# ---------------------------------------------------------------------------

class TestModelInfoEndpoint:

    def test_returns_200(self, client):
        assert client.get("/model/info").status_code == 200

    def test_response_schema(self, client):
        data = client.get("/model/info").json()
        assert set(data.keys()) == {
            "model_type", "version", "features", "training_date", "rmse", "description"
        }

    def test_metadata_values(self, client):
        data = client.get("/model/info").json()
        assert data["model_type"] == "KNeighborsRegressor"
        assert data["version"] == "1.0.0"
        assert isinstance(data["features"], list)
        assert len(data["features"]) > 0
        assert isinstance(data["rmse"], float)
        assert data["rmse"] > 0

    def test_returns_503_when_metadata_not_loaded(self):
        """
        GET /model/info must return 503 when metadata is unavailable.

        info.py imports get_metadata directly (`from app.services.model import
        get_metadata`), so the patch target must be the name bound in that
        module, not the original definition in app.services.model.
        """
        with patch("app.api.v1.routes.info.get_metadata", side_effect=RuntimeError("Metadata not loaded")):
            with TestClient(app, raise_server_exceptions=False) as c:
                response = c.get("/model/info")
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:

    def test_returns_200(self, client):
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200

    def test_response_schema(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert set(data.keys()) == {"predicted_price", "currency", "model_version"}

    def test_predicted_price_is_positive_float(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert isinstance(data["predicted_price"], float)
        assert data["predicted_price"] > 0

    def test_currency_is_usd(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["currency"] == "USD"

    def test_model_version_matches_metadata(self, client):
        prediction = client.post("/predict", json=VALID_PAYLOAD).json()
        info = client.get("/model/info").json()
        assert prediction["model_version"] == info["version"]

    def test_missing_field_returns_422(self, client):
        """Pydantic validation must reject requests missing required fields."""
        for missing_field in VALID_PAYLOAD:
            payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing_field}
            response = client.post("/predict", json=payload)
            assert response.status_code == 422, \
                f"Expected 422 when '{missing_field}' is missing, got {response.status_code}"

    def test_negative_bedrooms_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "bedrooms": -1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_zero_sqft_living_returns_422(self, client):
        """sqft_living has gt=0 constraint — zero must be rejected."""
        payload = {**VALID_PAYLOAD, "sqft_living": 0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_zero_floors_returns_422(self, client):
        """floors has gt=0 constraint — zero must be rejected."""
        payload = {**VALID_PAYLOAD, "floors": 0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_unknown_zipcode_returns_503(self, client):
        """
        A zipcode absent from the demographics table raises ValueError inside
        make_prediction(), which main.py catches under `except ValueError` → 503.
        """
        payload = {**VALID_PAYLOAD, "zipcode": "00000"}
        assert client.post("/predict", json=payload).status_code == 503

    def test_returns_500_when_model_not_loaded(self):
        """
        POST /predict returns 500 when get_model() raises RuntimeError.

        predict.py calls model_service.get_model(), which raises RuntimeError
        (never returns None). RuntimeError is not caught by the `except ValueError`
        branch in main.py, so it falls through to `except Exception` → 500.
        """
        with patch("app.services.model.get_model", side_effect=RuntimeError("Model not loaded")):
            with TestClient(app, raise_server_exceptions=False) as c:
                response = c.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 500

    def test_prediction_is_deterministic(self, client):
        """Same input must produce the same output (KNN is deterministic)."""
        first = client.post("/predict", json=VALID_PAYLOAD).json()["predicted_price"]
        second = client.post("/predict", json=VALID_PAYLOAD).json()["predicted_price"]
        assert first == second


# ---------------------------------------------------------------------------
# / (root)
# ---------------------------------------------------------------------------

class TestRootEndpoint:

    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_contains_endpoint_listing(self, client):
        data = client.get("/").json()
        assert "endpoints" in data
        assert isinstance(data["endpoints"], list)
        assert len(data["endpoints"]) > 0
