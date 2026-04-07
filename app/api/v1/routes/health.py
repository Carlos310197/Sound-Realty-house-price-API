from typing import Dict, Any
from app.services import model as model_service

def check_health() -> Dict[str, Any]:
    """
    Check the health status of the service.
    
    Returns:
        Dictionary with health status information
        
    This function should:
    1. Check if model is loaded (model is not None)
    2. Check if metadata is loaded (metadata is not None)
    3. Return a dictionary with status information
    """
    
    # Create health status dictionary
    try:
        model_loaded = model_service.get_model() is not None
    except RuntimeError:
        model_loaded = False

    try:
        metadata_loaded = model_service.get_metadata() is not None
    except RuntimeError:
        metadata_loaded = False
        
    # Determine overall health status
    is_healthy = model_loaded and metadata_loaded
    
    # Create a message based on which components are loaded
    if is_healthy:
        message = "Service is healthy and ready to make predictions"
    elif not model_loaded and not metadata_loaded:
        message = "Service is unhealthy: Model and metadata not loaded"
    elif not model_loaded:
        message = "Service is unhealthy: Model not loaded"
    else:
        message = "Service is unhealthy: Metadata not loaded"
    
    # Construct the health status dictionary
    health_status = {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": model_loaded,
        "message": message
    }
    
    return health_status