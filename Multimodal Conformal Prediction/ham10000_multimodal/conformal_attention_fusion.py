from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from conformal_prediction import (
    ClassificationScores,
    SplitConformalClassifier,
    evaluate_prediction_sets,
)
from .data import LABEL_NAMES
from .utils import classification_metrics, save_json, set_seed


SCORE_TYPES = ("probability", "cumulative", "high_probability")
VAL_SPLIT_ID = 0
TEST_SPLIT_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply split conformal prediction to the attention-based HAM10000 "
            "multimodal fusion model."
        )
    )
    parser.add_argument(
        "--attention-dir",
        required=True,
        help="Directory containing val_predictions.csv and test_predictions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write conformal outputs. Defaults inside attention-dir.",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--score-types", nargs="+", default=list(SCORE_TYPES), choices=SCORE_TYPES)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class PrefitProbabilityClassifier:
    """Tiny adapter so saved neural probabilities look like a fitted classifier."""

    def __init__(self, probabilities_by_split: dict[int, np.ndarray], classes: np.ndarray) -> None:
        self.probabilities_by_split = {
            int(split_id): np.asarray(probabilities, dtype=np.float64)
            for split_id, probabilities in probabilities_by_split.items()
        }
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must contain two columns: split_id and row_index.")

        split_ids = X[:, 0].astype(int)
        row_indices = X[:, 1].astype(int)
        probabilities = np.empty((len(X), len(self.classes_)), dtype=np.float64)

        for split_id in np.unique(split_ids):
            if split_id not in self.probabilities_by_split:
                raise ValueError(f"Unknown split id: {split_id}")
            mask = split_ids == split_id
            split_probabilities = self.probabilities_by_split[split_id]
            probabilities[mask] = split_probabilities[row_indices[mask]]

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    attention_dir = Path(args.attention_dir).expanduser().resolve()
    output_dir = resolve_output_dir(attention_dir, args.output_dir, args.alpha)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_names = sorted(LABEL_NAMES)
    target_names = [f"{label}: {LABEL_NAMES[label]}" for label in label_names]
    classes = np.arange(len(label_names), dtype=int)

    val_data = load_attention_predictions(attention_dir / "val_predictions.csv", label_names)
    test_data = load_attention_predictions(attention_dir / "test_predictions.csv", label_names)
    classifier = PrefitProbabilityClassifier(
        {
            VAL_SPLIT_ID: val_data["probabilities"],
            TEST_SPLIT_ID: test_data["probabilities"],
        },
        classes=classes,
    )

    X_calib = make_index_matrix(VAL_SPLIT_ID, len(val_data["labels"]))
    X_test = make_index_matrix(TEST_SPLIT_ID, len(test_data["labels"]))

    summary_rows = []
    summary_json: dict[str, Any] = {
        "alpha": float(args.alpha),
        "confidence_level": float(1.0 - args.alpha),
        "attention_dir": str(attention_dir),
        "calibration_split": "val",
        "test_split": "test",
        "score_types": list(args.score_types),
        "model_source": "saved_attention_fusion_probabilities",
        "conformal_classes": [
            "ClassificationScores",
            "SplitConformalClassifier",
            "evaluate_prediction_sets",
        ],
        "results": {},
    }

    for score_type in args.score_types:
        result = run_conformal_method(
            classifier=classifier,
            score_type=score_type,
            alpha=args.alpha,
            X_calib=X_calib,
            y_calib=val_data["labels"],
            X_test=X_test,
            y_test=test_data["labels"],
            test_rows=test_data["rows"],
            label_names=label_names,
            target_names=target_names,
            output_dir=output_dir,
        )
        summary_rows.append(result["summary_row"])
        summary_json["results"][score_type] = result["json"]

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / "conformal_summary.csv", index=False)
    save_json(summary_json, output_dir / "conformal_summary.json")

    print("\n========== HAM10000 CONFORMAL ATTENTION FUSION ==========")
    for row in summary_rows:
        print(
            f"{row['score_type']:>16} | "
            f"coverage={row['coverage']:.4f} | "
            f"avg_set_size={row['average_set_size']:.4f} | "
            f"q_hat={row['q_hat']:.4f}"
        )
    print(f"\nSaved attention conformal outputs to: {output_dir}")


def resolve_output_dir(attention_dir: Path, output_dir: str | None, alpha: float) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    alpha_tag = f"{alpha:.2f}".replace(".", "_")
    return attention_dir / f"conformal_alpha_{alpha_tag}"


def load_attention_predictions(path: Path, label_names: list[str]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find attention predictions file: {path}")

    frame = pd.read_csv(path)
    probability_columns = [f"probability_{label}" for label in label_names]
    required_columns = {"true_index", *probability_columns}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    probabilities = validate_probabilities(probabilities, path)
    labels = frame["true_index"].to_numpy(dtype=int)
    rows = frame[[column for column in ("image_id", "lesion_id", "dx") if column in frame.columns]].copy()
    return {
        "labels": labels,
        "probabilities": probabilities,
        "rows": rows if not rows.empty else None,
    }


def validate_probabilities(probabilities: np.ndarray, path: Path) -> np.ndarray:
    if probabilities.ndim != 2:
        raise ValueError(f"{path.name} probabilities must be a 2D array.")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{path.name} contains non-finite probabilities.")
    if np.any(probabilities < -1e-8):
        raise ValueError(f"{path.name} contains negative probabilities.")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(f"{path.name} contains probability rows with zero total mass.")
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        probabilities = probabilities / row_sums
    return probabilities


def make_index_matrix(split_id: int, size: int) -> np.ndarray:
    return np.column_stack(
        [
            np.full(size, split_id, dtype=int),
            np.arange(size, dtype=int),
        ]
    )


def run_conformal_method(
    classifier: PrefitProbabilityClassifier,
    score_type: str,
    alpha: float,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_rows: pd.DataFrame | None,
    label_names: list[str],
    target_names: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    cp = SplitConformalClassifier(
        model=classifier,
        alpha=alpha,
        score=ClassificationScores(score_type=score_type),
        prefit=True,
    )
    cp.calibrate(X_calib, y_calib)

    predicted = cp.predict(X_test).astype(int)
    probabilities = cp.predict_proba(X_test)
    prediction_sets = cp.predict_set(X_test)
    point_metrics = classification_metrics(y_test, predicted, target_names)
    set_metrics = evaluate_prediction_sets(y_test, prediction_sets, cp.classes_)

    save_prediction_outputs(
        score_type=score_type,
        y_true=y_test,
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
        "calibration_size": int(len(y_calib)),
        "test_size": int(len(y_test)),
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
        predictions = test_rows.copy()
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
