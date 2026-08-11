from fastapi import FastAPI, HTTPException

from app.schemas import (
    ClassificationPredictionRequest,
    ClassificationPredictionResponse,
)

from app.model_service import (
    load_model_registry,
    predict_single_classification_case,
)


app = FastAPI(
    title="Conformal Classification API",
    description="API for conformal prediction sets.",
    version="0.1.0"
)


model_registry = load_model_registry()


@app.get("/")
def root():
    return {
        "message": "Conformal Classification API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(model_registry.keys())
    }


@app.post(
    "/predict-classification",
    response_model=ClassificationPredictionResponse
)
def predict_classification(request: ClassificationPredictionRequest):
    try:
        features = request.model_dump()
        method = features.pop("method")

        result = predict_single_classification_case(
            model_registry=model_registry,
            method=method,
            features=features
        )

        return result

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
