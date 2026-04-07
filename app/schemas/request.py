from pydantic import BaseModel, Field

class HousePredictionRequest(BaseModel):
    """Schema for house prediction request.

    Accepts the 7 house features from the sales data plus zipcode.
    Demographics features are enriched internally via the demographics service.
    """

    #TODO - Add more realistic validation constraints (e.g. bedrooms >= 0, sqft_living > 0, etc.) using an EDA of the training data
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms (can be decimal, e.g. 2.5)")
    sqft_living: float = Field(..., gt=0, description="Interior living space in square feet")
    sqft_lot: float = Field(..., gt=0, description="Lot area in square feet")
    floors: float = Field(..., gt=0, description="Number of floors (can be decimal, e.g. 1.5)")
    sqft_above: float = Field(..., ge=0, description="Square footage above ground level")
    sqft_basement: float = Field(..., ge=0, description="Square footage of the basement")
    zipcode: str = Field(..., description="5-digit ZIP code used to enrich demographics features")

    class Config:
        json_schema_extra = {
            "example": {
                "bedrooms": 4,
                "bathrooms": 1.0,
                "sqft_living": 1680,
                "sqft_lot": 5043,
                "floors": 1.5,
                "sqft_above": 1680,
                "sqft_basement": 0,
                "zipcode": "98118"
            }
        }