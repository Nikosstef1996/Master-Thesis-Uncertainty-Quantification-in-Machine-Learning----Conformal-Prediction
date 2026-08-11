import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from app.regressor import SplitConformalRegressor
from app.scores import RegressionScores

# Import these from wherever you defined them
# Example:
# from your_conformal_module import SplitConformalRegressor, RegressionScores


configs = {
    "residual": {
        "score": RegressionScores(score_type="residual"),
        "kwargs": {
            "point_model": RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        }
    },
    "scaled_residual": {
        "score": RegressionScores(score_type="scaled_residual"),
        "kwargs": {
            "point_model": RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),
            "scale_model": RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        }
    },
    "cqr": {
        "score": RegressionScores(score_type="cqr"),
        "kwargs": {
            "lower_model": GradientBoostingRegressor(
                loss="quantile",
                alpha=0.05,
                random_state=42
            ),
            "upper_model": GradientBoostingRegressor(
                loss="quantile",
                alpha=0.95,
                random_state=42
            )
        }
    }
}

def train_all_models(X_train, y_train, X_calib, y_calib, X_test, y_test, alpha=0.1):
    results = {}

    for name, cfg in configs.items():
        cp = SplitConformalRegressor(
            score=cfg["score"],
            alpha=alpha,
            **cfg["kwargs"]
        )

        cp.fit(X_train, y_train, X_calib, y_calib)

        y_pred = cp.predict(X_test)
        lower, upper = cp.predict_interval(X_test)

        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        avg_width = np.mean(upper - lower)

        results[name] = {
            "cp": cp,
            "coverage": float(coverage),
            "avg_width": float(avg_width),
            "q_hat": float(cp.q_hat_),
            "alpha": alpha
        }

    return results

def save_model_registry(results, path="models/housing_cp_models.joblib"):
    joblib.dump(results, path)


def load_model_registry(path="models/housing_cp_models.joblib"):
    return joblib.load(path) ## It loads the results dictionary including the prediction interval and the class.


FEATURE_ORDER = [
    "median_income",
    "house_age",
    "average_rooms",
    "average_bedrooms",
    "population",
    "average_occupancy",
    "latitude",
    "longitude",
]


def predict_single_housing_case(model_registry, method, features):
    if method not in model_registry:
        allowed = list(model_registry.keys())
        raise ValueError(f"Unknown method '{method}'. Allowed methods: {allowed}")

    cp = model_registry[method]["cp"]

    X_new = np.array([[
        features["median_income"],
        features["house_age"],
        features["average_rooms"],
        features["average_bedrooms"],
        features["population"],
        features["average_occupancy"],
        features["latitude"],
        features["longitude"],
    ]])

    y_pred = cp.predict(X_new)
    lower, upper = cp.predict_interval(X_new)

    alpha = model_registry[method]["alpha"]

    return {
        "method": method,
        "point_prediction": float(y_pred[0]),
        "prediction_interval": {
            "lower": float(lower[0]),
            "upper": float(upper[0])
        },
        "confidence_level": float(1 - alpha),
        "q_hat": float(model_registry[method]["q_hat"]),
        "test_coverage": float(model_registry[method]["coverage"]),
        "average_interval_width": float(model_registry[method]["avg_width"]),
    }
