from typing import Dict, List

from pydantic import BaseModel, Field


class ClassificationPredictionRequest(BaseModel):
    method: str = Field(
        default="probability",
        description="One of: probability, cumulative, high_probability"
    )

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class ClassificationPredictionResponse(BaseModel):
    method: str
    predicted_class: str
    prediction_set: List[str]
    probabilities: Dict[str, float]
    confidence_level: float
    q_hat: float
    test_coverage: float
    average_set_size: float