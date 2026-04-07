from fastapi import FastAPI, HTTPException

from app.schemas.request import HousePredictionRequest
from app.schemas.response import PredictionResponse, ModelInfoResponse, HealthCheckResponse

from app.api.v1.routes import health, info, predict as predict_route
from app.services.model import load_model_and_metadata
from app.services.demographics import load_demographics
from app.core.config import settings
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Machine learning service for predicting house prices in Seattle for Sound Realty",
    version=settings.app_version,
)

# Load model at startup
@app.on_event("startup")
async def startup_event():
    """Load model and metadata when the service starts"""
    try:
        load_model_and_metadata()
    except Exception as e:
        logger.warning("Failed to load model at startup: %s", e)

    try:
        load_demographics()
    except Exception as e:
        logger.warning("Failed to load demographics data at startup: %s", e)

# Implement the health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Check if the service is healthy and model is loaded.
    
    Returns:
        HealthCheckResponse with current service status
    """
    # Get health status from api.py
    health_status = health.check_health()
    
    # Create and return HealthCheckResponse
    return HealthCheckResponse(
        status=health_status["status"],
        model_loaded=health_status["model_loaded"],
        message=health_status["message"]
    )

# Implement the model info endpoint
@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        ModelInfoResponse with model metadata
        
    Raises:
        HTTPException: If model metadata is not loaded
    """
    try:
        # Get model info and return it
        info_data = info.get_model_info()
        return ModelInfoResponse(
            model_type=info_data["model_type"],
            version=info_data["version"],
            features=info_data["features"],
            training_date=info_data["training_date"],
            rmse=info_data["rmse"],
            description=info_data["description"]
        )
    except ValueError as e:
        # Raise HTTPException with status_code=503 (Service Unavailable)
        raise HTTPException(
            status_code=503, 
            detail=f"Model information not available: {str(e)}"
        )

# Implement the prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    """
    Predict house price based on property features.
    
    Args:
        request: HousePredictionRequest with all 13 property features
        
    Returns:
        PredictionResponse with predicted price
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    try:
        # Convert request to dictionary
        features_dict = request.model_dump()
        
        # Make prediction
        predicted_price = predict_route.make_prediction(features_dict)
        
        # Get model version from metadata
        model_info = info.get_model_info()
        model_version = model_info["version"]
        
        # Create and return PredictionResponse
        return PredictionResponse(
            predicted_price=predicted_price,
            currency="USD",
            model_version=model_version
        )
        
    except ValueError as e:
        # Model not loaded error
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Other errors
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Root endpoint for basic information
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "House Price Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/health - Check service health",
            "/model/info - Get model information",
            "/predict - Make price prediction",
            "/docs - Interactive API documentation"
        ]
    }
    