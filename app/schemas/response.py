from pydantic import BaseModel, Field
from typing import List

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    predicted_price: float = Field(..., description="Predicted house price in USD")
    currency: str = Field(default="USD", description="Currency of the predicted price")
    model_version: str = Field(..., description="Version of the model used for prediction")

class ModelInfoResponse(BaseModel):
    """Schema for model information response"""
    model_type: str = Field(..., description="Type of machine learning model")
    version: str = Field(..., description="Model version number")
    features: List[str] = Field(..., description="List of feature names expected by the model")
    training_date: str = Field(..., description="Date when the model was trained")
    rmse: float = Field(..., description="Root Mean Squared Error from model training")
    description: str = Field(..., description="Description of what the model does")

class HealthCheckResponse(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(..., description="Whether the model is successfully loaded")
    message: str = Field(..., description="Descriptive message about service status")