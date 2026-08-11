from pydantic import BaseModel, Field


class HousingPredictionRequest(BaseModel):
    method: str = Field(default="cqr", description="One of: residual, scaled_residual, cqr")

    median_income: float
    house_age: float
    average_rooms: float
    average_bedrooms: float
    population: float
    average_occupancy: float
    latitude: float
    longitude: float


class PredictionInterval(BaseModel):
    lower: float
    upper: float


class HousingPredictionResponse(BaseModel):
    method: str
    point_prediction: float
    prediction_interval: PredictionInterval
    confidence_level: float
    q_hat: float
    test_coverage: float
    average_interval_width: float
