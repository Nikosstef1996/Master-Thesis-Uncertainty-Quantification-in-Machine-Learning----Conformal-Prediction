from fastapi import FastAPI, HTTPException

from app.schemas import HousingPredictionRequest, HousingPredictionResponse
from app.model_service import load_model_registry, predict_single_housing_case


app = FastAPI(
    title="Conformal Housing Price API",
    description="API for conformal prediction intervals on American housing data.",
    version="0.1.0"
)

model_registry = load_model_registry() ## Loads results.


@app.get("/")
def root():
    return {
        "message": "Conformal Housing Price API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(model_registry.keys())
    }


@app.post("/predict-housing-price", response_model=HousingPredictionResponse)
def predict_housing_price(request: HousingPredictionRequest):
    try:
        features = request.model_dump()
        method = features.pop("method") 

        result = predict_single_housing_case(
            model_registry=model_registry,
            method=method,
            features=features
        )

        return result

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
