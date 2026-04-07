"""
Unit tests for request and response Pydantic schemas.

These tests instantiate models directly — no HTTP, no mocks.
They verify that field constraints are enforced at the model level,
independently of the FastAPI layer.
"""
import pytest
from pydantic import ValidationError

from app.schemas.request import HousePredictionRequest
from app.schemas.response import PredictionResponse, ModelInfoResponse, HealthCheckResponse


VALID_INPUT = {
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 1500,
    "sqft_lot": 4000,
    "floors": 1.0,
    "sqft_above": 1500,
    "sqft_basement": 0,
    "zipcode": "98042",
}


# ---------------------------------------------------------------------------
# HousePredictionRequest — valid construction
# ---------------------------------------------------------------------------

class TestHousePredictionRequestValid:

    def test_accepts_valid_input(self):
        req = HousePredictionRequest(**VALID_INPUT)
        assert req.bedrooms == 3
        assert req.zipcode == "98042"

    def test_accepts_zero_bedrooms(self):
        """bedrooms has ge=0 — zero is allowed."""
        req = HousePredictionRequest(**{**VALID_INPUT, "bedrooms": 0})
        assert req.bedrooms == 0

    def test_accepts_zero_sqft_basement(self):
        """sqft_basement has ge=0 — zero is allowed (no basement)."""
        req = HousePredictionRequest(**{**VALID_INPUT, "sqft_basement": 0})
        assert req.sqft_basement == 0

    def test_accepts_decimal_bathrooms(self):
        req = HousePredictionRequest(**{**VALID_INPUT, "bathrooms": 2.5})
        assert req.bathrooms == 2.5

    def test_accepts_decimal_floors(self):
        req = HousePredictionRequest(**{**VALID_INPUT, "floors": 1.5})
        assert req.floors == 1.5


# ---------------------------------------------------------------------------
# HousePredictionRequest — constraint violations
# ---------------------------------------------------------------------------

class TestHousePredictionRequestInvalid:

    def test_rejects_negative_bedrooms(self):
        with pytest.raises(ValidationError):
            HousePredictionRequest(**{**VALID_INPUT, "bedrooms": -1})

    def test_rejects_negative_bathrooms(self):
        with pytest.raises(ValidationError):
            HousePredictionRequest(**{**VALID_INPUT, "bathrooms": -0.5})

    def test_rejects_zero_sqft_living(self):
        """sqft_living has gt=0 — zero must be rejected."""
        with pytest.raises(ValidationError):
            HousePredictionRequest(**{**VALID_INPUT, "sqft_living": 0})

    def test_rejects_zero_sqft_lot(self):
        with pytest.raises(ValidationError):
            HousePredictionRequest(**{**VALID_INPUT, "sqft_lot": 0})

    def test_rejects_zero_floors(self):
        with pytest.raises(ValidationError):
            HousePredictionRequest(**{**VALID_INPUT, "floors": 0})

    def test_rejects_missing_zipcode(self):
        data = {k: v for k, v in VALID_INPUT.items() if k != "zipcode"}
        with pytest.raises(ValidationError):
            HousePredictionRequest(**data)


# ---------------------------------------------------------------------------
# Response schemas — valid construction
# ---------------------------------------------------------------------------

class TestResponseSchemas:

    def test_prediction_response(self):
        r = PredictionResponse(predicted_price=450000.0, currency="USD", model_version="1.0.0")
        assert r.predicted_price == 450000.0
        assert r.currency == "USD"

    def test_model_info_response(self):
        r = ModelInfoResponse(
            model_type="KNeighborsRegressor",
            version="1.0.0",
            features=["bedrooms", "sqft_living"],
            training_date="2026-04-05",
            rmse=201659.43,
            description="KNN model",
        )
        assert r.rmse == 201659.43
        assert len(r.features) == 2

    def test_health_check_response(self):
        r = HealthCheckResponse(status="healthy", model_loaded=True, message="OK")
        assert r.status == "healthy"
        assert r.model_loaded is True
