from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from conformal_prediction import (
    ClassificationScores,
    SplitConformalClassifier,
    evaluate_prediction_sets,
)
from .data import LABEL_NAMES
from .utils import classification_metrics, save_json, set_seed


SCORE_TYPES = ("probability", "cumulative", "high_probability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply split conformal prediction to the multimodal HAM10000 "
            "Random Forest classifier."
        )
    )
    parser.add_argument(
        "--features-npz",
        required=True,
        help="Path to fused_features.npz from ham10000-random-forest.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write conformal outputs. Defaults next to features-npz.",
    )
    parser.add_argument(
        "--model-joblib",
        default=None,
        help=(
            "Optional fitted Random Forest joblib. Defaults to random_forest.joblib "
            "next to features-npz when it exists."
        ),
    )
    parser.add_argument(
        "--splits-csv",
        default=None,
        help="Optional splits.csv so prediction files include image_id and lesion_id.",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--score-types", nargs="+", default=list(SCORE_TYPES), choices=SCORE_TYPES)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train a new Random Forest instead of loading an existing fitted model.",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--class-weight", default="balanced", choices=("balanced", "balanced_subsample", "none"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    features_path = Path(args.features_npz).expanduser().resolve()
    output_dir = resolve_output_dir(features_path, args.output_dir, args.alpha)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_feature_arrays(features_path)
    label_names = sorted(LABEL_NAMES)
    target_names = [f"{label}: {LABEL_NAMES[label]}" for label in label_names]

    classifier, model_source = load_or_train_classifier(args, features_path, arrays)
    test_rows = load_split_rows(args.splits_csv, features_path, split_name="test")

    summary_rows = []
    summary_json: dict[str, Any] = {
        "alpha": float(args.alpha),
        "confidence_level": float(1.0 - args.alpha),
        "features_npz": str(features_path),
        "model_source": model_source,
        "calibration_split": "val",
        "test_split": "test",
        "score_types": list(args.score_types),
        "results": {},
    }

    for score_type in args.score_types:
        result = run_conformal_method(
            classifier=classifier,
            score_type=score_type,
            alpha=args.alpha,
            arrays=arrays,
            label_names=label_names,
            target_names=target_names,
            test_rows=test_rows,
            output_dir=output_dir,
        )
        summary_rows.append(result["summary_row"])
        summary_json["results"][score_type] = result["json"]

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / "conformal_summary.csv", index=False)
    save_json(summary_json, output_dir / "conformal_summary.json")

    print("\n========== HAM10000 CONFORMAL RANDOM FOREST ==========")
    for row in summary_rows:
        print(
            f"{row['score_type']:>16} | "
            f"coverage={row['coverage']:.4f} | "
            f"avg_set_size={row['average_set_size']:.4f} | "
            f"q_hat={row['q_hat']:.4f}"
        )
    print(f"\nSaved conformal outputs to: {output_dir}")


def resolve_output_dir(features_path: Path, output_dir: str | None, alpha: float) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    alpha_tag = f"{alpha:.2f}".replace(".", "_")
    return features_path.parent / f"conformal_alpha_{alpha_tag}"


def load_feature_arrays(features_path: Path) -> dict[str, np.ndarray]:
    if not features_path.exists():
        raise FileNotFoundError(f"Could not find features file: {features_path}")

    with np.load(features_path, allow_pickle=False) as data:
        required = {
            "train_features",
            "train_labels",
            "val_features",
            "val_labels",
            "test_features",
            "test_labels",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"features-npz is missing arrays: {sorted(missing)}")
        return {key: data[key] for key in data.files}


def load_or_train_classifier(
    args: argparse.Namespace,
    features_path: Path,
    arrays: dict[str, np.ndarray],
) -> tuple[RandomForestClassifier, str]:
    model_joblib = Path(args.model_joblib).expanduser().resolve() if args.model_joblib else None
    if model_joblib is None:
        candidate = features_path.parent / "random_forest.joblib"
        model_joblib = candidate if candidate.exists() else None

    if not args.train_model and model_joblib is not None and model_joblib.exists():
        payload = joblib.load(model_joblib)
        if isinstance(payload, dict) and "classifier" in payload:
            classifier = payload["classifier"]
        else:
            classifier = payload
        if not hasattr(classifier, "predict_proba") or not hasattr(classifier, "classes_"):
            raise TypeError(f"Loaded object is not a fitted probabilistic classifier: {model_joblib}")
        return classifier, str(model_joblib)

    class_weight = None if args.class_weight == "none" else args.class_weight
    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight=class_weight,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
    print("Training RandomForestClassifier for conformal prediction...")
    classifier.fit(arrays["train_features"], arrays["train_labels"])
    return classifier, "trained_from_features"


def load_split_rows(splits_csv: str | None, features_path: Path, split_name: str) -> pd.DataFrame | None:
    split_path = Path(splits_csv).expanduser().resolve() if splits_csv else features_path.parent / "splits.csv"
    if not split_path.exists():
        return None
    split_frame = pd.read_csv(split_path)
    required = {"split", "image_id", "lesion_id", "dx"}
    if not required.issubset(split_frame.columns):
        return None
    return split_frame[split_frame["split"] == split_name].reset_index(drop=True)


def run_conformal_method(
    classifier: RandomForestClassifier,
    score_type: str,
    alpha: float,
    arrays: dict[str, np.ndarray],
    label_names: list[str],
    target_names: list[str],
    test_rows: pd.DataFrame | None,
    output_dir: Path,
) -> dict[str, Any]:
    cp = SplitConformalClassifier(
        model=classifier,
        alpha=alpha,
        score=ClassificationScores(score_type=score_type),
        prefit=True,
    )
    cp.calibrate(arrays["val_features"], arrays["val_labels"])

    predicted = cp.predict(arrays["test_features"]).astype(int)
    probabilities = cp.predict_proba(arrays["test_features"])
    prediction_sets = cp.predict_set(arrays["test_features"])
    point_metrics = classification_metrics(arrays["test_labels"], predicted, target_names)
    set_metrics = evaluate_prediction_sets(arrays["test_labels"], prediction_sets, cp.classes_)

    save_prediction_outputs(
        score_type=score_type,
        y_true=arrays["test_labels"],
        predicted=predicted,
        probabilities=probabilities,
        prediction_sets=prediction_sets,
        set_metrics=set_metrics,
        label_names=label_names,
        test_rows=test_rows,
        output_dir=output_dir,
    )
    save_class_coverage(score_type, set_metrics, label_names, output_dir)
    pd.DataFrame({"calibration_score": cp.calibration_scores_}).to_csv(
        output_dir / f"{score_type}_calibration_scores.csv",
        index=False,
    )

    summary_row = {
        "score_type": score_type,
        "alpha": float(alpha),
        "confidence_level": float(1.0 - alpha),
        "q_hat": float(cp.q_hat_),
        "calibration_size": int(arrays["val_labels"].shape[0]),
        "test_size": int(arrays["test_labels"].shape[0]),
        "point_accuracy": point_metrics["accuracy"],
        "point_balanced_accuracy": point_metrics["balanced_accuracy"],
        "coverage": set_metrics["coverage"],
        "average_set_size": set_metrics["average_set_size"],
        "median_set_size": set_metrics["median_set_size"],
        "singleton_fraction": set_metrics["singleton_fraction"],
        "max_set_size": set_metrics["max_set_size"],
    }
    json_result = {
        **summary_row,
        "per_class": convert_per_class_labels(set_metrics["per_class"], label_names),
    }
    return {"summary_row": summary_row, "json": json_result}


def save_prediction_outputs(
    score_type: str,
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    prediction_sets: list[np.ndarray],
    set_metrics: dict[str, Any],
    label_names: list[str],
    test_rows: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    if test_rows is not None and len(test_rows) == len(y_true):
        predictions = test_rows[["image_id", "lesion_id", "dx"]].copy()
    else:
        predictions = pd.DataFrame({"row_index": np.arange(len(y_true))})

    predictions["true_index"] = y_true
    predictions["true_label"] = [label_names[int(index)] for index in y_true]
    predictions["predicted_index"] = predicted
    predictions["predicted_label"] = [label_names[int(index)] for index in predicted]
    predictions["covered"] = set_metrics["covered"]
    predictions["prediction_set_size"] = set_metrics["set_sizes"]
    predictions["prediction_set"] = [
        ";".join(label_names[int(index)] for index in pred_set)
        for pred_set in prediction_sets
    ]
    predictions["predicted_probability"] = probabilities[np.arange(len(predicted)), predicted]
    predictions["true_probability"] = probabilities[np.arange(len(y_true)), y_true]
    for index, label in enumerate(label_names):
        predictions[f"probability_{label}"] = probabilities[:, index]

    predictions.to_csv(output_dir / f"{score_type}_test_predictions.csv", index=False)


def save_class_coverage(
    score_type: str,
    set_metrics: dict[str, Any],
    label_names: list[str],
    output_dir: Path,
) -> None:
    rows = []
    for class_index, values in set_metrics["per_class"].items():
        label = label_names[int(class_index)]
        rows.append(
            {
                "class_index": int(class_index),
                "class_label": label,
                "support": values["support"],
                "coverage": values["coverage"],
                "average_set_size": values["average_set_size"],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / f"{score_type}_coverage_by_class.csv", index=False)


def convert_per_class_labels(per_class: dict[str, Any], label_names: list[str]) -> dict[str, Any]:
    converted = {}
    for class_index, values in per_class.items():
        label = label_names[int(class_index)]
        converted[label] = values
    return converted


if __name__ == "__main__":
    main()
