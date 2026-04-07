from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    app_name: str = "House Price Prediction API for Sound Realty"
    app_version: str = "1.0.0"
    environment: str = "development"

    # Paths
    model_path: str = "model/model.pkl"
    model_metadata_path: str = "model/model_metadata.json"
    demographics_path: str = "data/zipcode_demographics.csv"

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
