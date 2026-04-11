from pydantic import BaseModel, Field


class HousePredictionRequest(BaseModel):
    """Minimal prediction request — only the features the model requires.

    Accepts the 7 house features the model actually uses plus zipcode.
    Demographics features are enriched internally via the demographics service.
    """

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


class FullHousePredictionRequest(BaseModel):
    """Full prediction request — all columns from future_unseen_examples.csv.

    Accepts every attribute a caller would have from the house-sales dataset.
    The service extracts only the features the model needs and enriches
    demographics via zipcode on the backend.
    """

    bedrooms: int = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms (can be decimal, e.g. 2.5)")
    sqft_living: float = Field(..., gt=0, description="Interior living space in square feet")
    sqft_lot: float = Field(..., gt=0, description="Lot area in square feet")
    floors: float = Field(..., gt=0, description="Number of floors (can be decimal, e.g. 1.5)")
    waterfront: int = Field(..., ge=0, le=1, description="Whether the property has a waterfront view (0 or 1)")
    view: int = Field(..., ge=0, le=4, description="Quality of view from the property (0–4)")
    condition: int = Field(..., ge=1, le=5, description="Overall condition of the house (1–5)")
    grade: int = Field(..., ge=1, description="Overall grade given to the housing unit based on King County grading system")
    sqft_above: float = Field(..., ge=0, description="Square footage above ground level")
    sqft_basement: float = Field(..., ge=0, description="Square footage of the basement")
    yr_built: int = Field(..., ge=1800, description="Year the house was built")
    yr_renovated: int = Field(..., ge=0, description="Year the house was renovated (0 if never renovated)")
    zipcode: str = Field(..., description="5-digit ZIP code used to enrich demographics features")
    lat: float = Field(..., description="Latitude coordinate of the property")
    long: float = Field(..., description="Longitude coordinate of the property")
    sqft_living15: float = Field(..., gt=0, description="Interior living space of the nearest 15 neighbors (sq ft)")
    sqft_lot15: float = Field(..., gt=0, description="Lot area of the nearest 15 neighbors (sq ft)")

    class Config:
        json_schema_extra = {
            "example": {
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
                "zipcode": "98118",
                "lat": 47.5354,
                "long": -122.273,
                "sqft_living15": 1560,
                "sqft_lot15": 5765
            }
        }
