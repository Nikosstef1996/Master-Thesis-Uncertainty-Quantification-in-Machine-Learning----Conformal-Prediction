import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

from app.classifier import SplitConformalClassifier
from app.scores import ClassificationScores


MODEL_PATH = "models/classification_cp_models.joblib"


configs = {
    "probability": {
        "score": ClassificationScores(score_type="probability"),
        "model": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    },
    "cumulative": {
        "score": ClassificationScores(score_type="cumulative"),
        "model": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    },
    "high_probability": {
        "score": ClassificationScores(score_type="high_probability"),
        "model": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }
}


FEATURE_ORDER = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def train_all_models(
    X_train,
    y_train,
    X_calib,
    y_calib,
    X_test,
    y_test,
    target_names=None,
    alpha=0.1
):
    results = {}

    for name, cfg in configs.items():
        cp = SplitConformalClassifier(
            model=cfg["model"],
            alpha=alpha,
            score=cfg["score"]
        )

        cp.fit(X_train, y_train, X_calib, y_calib)

        pred_probs = cp.predict_proba(X_test)
        pred_sets = cp.predict_set(X_test)
        point_preds = cp.predict(X_test)

        covered = np.array([
            y_test[i] in pred_sets[i]
            for i in range(len(y_test))
        ])

        set_sizes = np.array([
            len(pred_sets[i])
            for i in range(len(pred_sets))
        ])

        results[name] = {
            "cp": cp,
            "q_hat": float(cp.q_hat_),
            "alpha": float(alpha),
            "coverage": float(covered.mean()),
            "avg_set_size": float(set_sizes.mean()),
            "classes": list(cp.classes_),
            "target_names": list(target_names) if target_names is not None else None,
        }

    return results


def save_model_registry(results, path=MODEL_PATH):
    joblib.dump(results, path)


def load_model_registry(path=MODEL_PATH):
    return joblib.load(path)


def predict_single_classification_case(model_registry, method, features):
    if method not in model_registry:
        allowed = list(model_registry.keys())
        raise ValueError(f"Unknown method '{method}'. Allowed methods: {allowed}")

    cp = model_registry[method]["cp"]

    X_new = np.array([[
        features["sepal_length"],
        features["sepal_width"],
        features["petal_length"],
        features["petal_width"],
    ]])

    probs = cp.predict_proba(X_new)[0]
    point_pred = cp.predict(X_new)[0]
    pred_set = cp.predict_set(X_new)[0]

    classes = model_registry[method]["classes"]
    target_names = model_registry[method]["target_names"]

    probabilities = {}

    for class_label, probability in zip(classes, probs):
        if target_names is not None:
            display_label = target_names[int(class_label)]
        else:
            display_label = str(class_label)

        probabilities[display_label] = float(probability)

    if target_names is not None:
        predicted_class = target_names[int(point_pred)]
        prediction_set = [
            target_names[int(label)]
            for label in pred_set
        ]
    else:
        predicted_class = str(point_pred)
        prediction_set = [
            str(label)
            for label in pred_set
        ]

    alpha = model_registry[method]["alpha"]

    return {
        "method": method,
        "predicted_class": predicted_class,
        "prediction_set": prediction_set,
        "probabilities": probabilities,
        "confidence_level": float(1 - alpha),
        "q_hat": float(model_registry[method]["q_hat"]),
        "test_coverage": float(model_registry[method]["coverage"]),
        "average_set_size": float(model_registry[method]["avg_set_size"]),
    }
