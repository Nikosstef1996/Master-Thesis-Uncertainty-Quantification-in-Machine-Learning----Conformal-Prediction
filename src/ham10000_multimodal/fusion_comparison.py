from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .data import LABEL_NAMES
from .random_forest import predict_proba_full
from .utils import save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare HAM10000 multimodal fusion strategies using saved "
            "ResNet image features and metadata features."
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
        help="Where to save fusion comparison outputs.",
    )
    parser.add_argument(
        "--splits-csv",
        default=None,
        help="Optional splits.csv to include image IDs in prediction files.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--class-weight", default="balanced", choices=("balanced", "balanced_subsample", "none"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--late-fusion-metric",
        default="balanced_accuracy",
        choices=("balanced_accuracy", "accuracy", "macro_f1"),
        help="Validation metric used to tune the late-fusion image weight.",
    )
    parser.add_argument(
        "--late-fusion-step",
        type=float,
        default=0.05,
        help="Grid step for tuned late fusion image weight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    features_path = Path(args.features_npz).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else features_path.parent / "fusion_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_feature_arrays(features_path)
    feature_names = [str(name) for name in arrays["feature_names"]]
    feature_groups = get_feature_groups(feature_names)
    split_features = make_feature_views(arrays, feature_groups)

    label_names = sorted(LABEL_NAMES)
    target_names = [f"{label}: {LABEL_NAMES[label]}" for label in label_names]
    num_classes = len(label_names)
    split_rows = load_split_rows(args.splits_csv, features_path)

    classifier_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "class_weight": None if args.class_weight == "none" else args.class_weight,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        "verbose": args.verbose,
    }

    print("Training image-only Random Forest...")
    image_model = build_classifier(classifier_params)
    image_model.fit(split_features["train"]["image"], arrays["train_labels"])

    print("Training metadata-only Random Forest...")
    metadata_model = build_classifier(classifier_params)
    metadata_model.fit(split_features["train"]["metadata"], arrays["train_labels"])

    print("Training concatenation-fusion Random Forest...")
    concat_model = build_classifier(classifier_params)
    concat_model.fit(split_features["train"]["fusion"], arrays["train_labels"])

    image_val_probs = predict_proba_full(image_model, split_features["val"]["image"], num_classes)
    metadata_val_probs = predict_proba_full(metadata_model, split_features["val"]["metadata"], num_classes)
    image_test_probs = predict_proba_full(image_model, split_features["test"]["image"], num_classes)
    metadata_test_probs = predict_proba_full(metadata_model, split_features["test"]["metadata"], num_classes)

    weight_search = tune_late_fusion_weight(
        y_true=arrays["val_labels"],
        image_probs=image_val_probs,
        metadata_probs=metadata_val_probs,
        step=args.late_fusion_step,
        metric=args.late_fusion_metric,
    )
    weight_search.to_csv(output_dir / "late_fusion_weight_search.csv", index=False)
    best_weight = float(weight_search.iloc[0]["image_weight"])

    strategies = [
        StrategyResult(
            name="image_only",
            display_name="Image-only RF",
            val_probs=image_val_probs,
            test_probs=image_test_probs,
        ),
        StrategyResult(
            name="metadata_only",
            display_name="Metadata-only RF",
            val_probs=predict_proba_full(metadata_model, split_features["val"]["metadata"], num_classes),
            test_probs=metadata_test_probs,
        ),
        StrategyResult(
            name="feature_concatenation",
            display_name="Feature concatenation RF",
            val_probs=predict_proba_full(concat_model, split_features["val"]["fusion"], num_classes),
            test_probs=predict_proba_full(concat_model, split_features["test"]["fusion"], num_classes),
        ),
        StrategyResult(
            name="late_fusion_equal",
            display_name="Late fusion RF, equal weights",
            val_probs=late_fusion_probs(image_val_probs, metadata_val_probs, image_weight=0.5),
            test_probs=late_fusion_probs(image_test_probs, metadata_test_probs, image_weight=0.5),
            image_weight=0.5,
        ),
        StrategyResult(
            name="late_fusion_tuned",
            display_name=f"Late fusion RF, tuned {args.late_fusion_metric}",
            val_probs=late_fusion_probs(image_val_probs, metadata_val_probs, image_weight=best_weight),
            test_probs=late_fusion_probs(image_test_probs, metadata_test_probs, image_weight=best_weight),
            image_weight=best_weight,
        ),
    ]

    summary_rows = []
    json_results: dict[str, Any] = {}
    for strategy in strategies:
        for split_name in ("val", "test"):
            probabilities = strategy.val_probs if split_name == "val" else strategy.test_probs
            labels = arrays[f"{split_name}_labels"]
            predictions = probabilities.argmax(axis=1)
            metrics = compute_metrics(labels, predictions, target_names)
            summary_rows.append(
                {
                    "strategy": strategy.name,
                    "display_name": strategy.display_name,
                    "split": split_name,
                    "image_weight": strategy.image_weight,
                    "metadata_weight": None if strategy.image_weight is None else 1.0 - strategy.image_weight,
                    **metrics["summary"],
                }
            )
            json_results.setdefault(strategy.name, {})[split_name] = metrics["json"]
            save_split_outputs(
                output_dir=output_dir,
                strategy=strategy,
                split_name=split_name,
                probabilities=probabilities,
                labels=labels,
                label_names=label_names,
                target_names=target_names,
                rows=split_rows.get(split_name),
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "fusion_summary.csv", index=False)
    save_json(
        {
            "features_npz": str(features_path),
            "splits_csv": str(resolve_split_path(args.splits_csv, features_path)),
            "feature_groups": {
                "image_feature_count": len(feature_groups["image"]),
                "metadata_feature_count": len(feature_groups["metadata"]),
                "metadata_feature_names": [feature_names[index] for index in feature_groups["metadata"]],
            },
            "classifier_params": classifier_params,
            "late_fusion_metric": args.late_fusion_metric,
            "best_late_fusion_image_weight": best_weight,
            "results": json_results,
        },
        output_dir / "fusion_summary.json",
    )
    joblib.dump(
        {
            "image_only": image_model,
            "metadata_only": metadata_model,
            "feature_concatenation": concat_model,
            "best_late_fusion_image_weight": best_weight,
            "classifier_params": classifier_params,
            "feature_groups": feature_groups,
            "feature_names": feature_names,
        },
        output_dir / "fusion_models.joblib",
    )

    print("\n========== HAM10000 FUSION STRATEGY COMPARISON ==========")
    test_summary = summary[summary["split"] == "test"].sort_values("balanced_accuracy", ascending=False)
    for _, row in test_summary.iterrows():
        weight = "" if pd.isna(row["image_weight"]) else f" image_weight={row['image_weight']:.2f}"
        print(
            f"{row['display_name']:<36} | "
            f"acc={row['accuracy']:.4f} | "
            f"balanced_acc={row['balanced_accuracy']:.4f} | "
            f"macro_f1={row['macro_f1']:.4f}{weight}"
        )
    print(f"\nSaved fusion comparison outputs to: {output_dir}")


class StrategyResult:
    def __init__(
        self,
        name: str,
        display_name: str,
        val_probs: np.ndarray,
        test_probs: np.ndarray,
        image_weight: float | None = None,
    ) -> None:
        self.name = name
        self.display_name = display_name
        self.val_probs = val_probs
        self.test_probs = test_probs
        self.image_weight = image_weight


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
            "feature_names",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"features-npz is missing arrays: {sorted(missing)}")
        return {key: data[key] for key in data.files}


def get_feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    image = [index for index, name in enumerate(feature_names) if name.startswith("image_feature_")]
    metadata = [index for index, name in enumerate(feature_names) if not name.startswith("image_feature_")]
    if not image:
        raise ValueError("No image_feature_* columns were found.")
    if not metadata:
        raise ValueError("No metadata feature columns were found.")
    return {"image": image, "metadata": metadata, "fusion": list(range(len(feature_names)))}


def make_feature_views(
    arrays: dict[str, np.ndarray],
    feature_groups: dict[str, list[int]],
) -> dict[str, dict[str, np.ndarray]]:
    views: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("train", "val", "test"):
        features = arrays[f"{split_name}_features"]
        views[split_name] = {
            "image": features[:, feature_groups["image"]],
            "metadata": features[:, feature_groups["metadata"]],
            "fusion": features[:, feature_groups["fusion"]],
        }
    return views


def build_classifier(params: dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(**params)


def late_fusion_probs(
    image_probs: np.ndarray,
    metadata_probs: np.ndarray,
    image_weight: float,
) -> np.ndarray:
    return (image_weight * image_probs) + ((1.0 - image_weight) * metadata_probs)


def tune_late_fusion_weight(
    y_true: np.ndarray,
    image_probs: np.ndarray,
    metadata_probs: np.ndarray,
    step: float,
    metric: str,
) -> pd.DataFrame:
    if step <= 0.0 or step > 1.0:
        raise ValueError("late-fusion-step must be in (0, 1].")
    weights = np.arange(0.0, 1.0 + step / 2.0, step)
    rows = []
    for weight in weights:
        probs = late_fusion_probs(image_probs, metadata_probs, float(weight))
        pred = probs.argmax(axis=1)
        rows.append(
            {
                "image_weight": float(weight),
                "metadata_weight": float(1.0 - weight),
                "accuracy": float(accuracy_score(y_true, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
                "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values([metric, "balanced_accuracy", "accuracy"], ascending=False).reset_index(drop=True)


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    target_names: list[str],
) -> dict[str, Any]:
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    summary = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }
    return {"summary": summary, "json": {**summary, "report": report}}


def save_split_outputs(
    output_dir: Path,
    strategy: StrategyResult,
    split_name: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    target_names: list[str],
    rows: pd.DataFrame | None,
) -> None:
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(label_names))))
    pd.DataFrame(matrix, index=label_names, columns=label_names).to_csv(
        output_dir / f"{strategy.name}_{split_name}_confusion_matrix.csv"
    )
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).T.to_csv(output_dir / f"{strategy.name}_{split_name}_classification_report.csv")

    if rows is not None and len(rows) == len(labels):
        predictions_frame = rows[["image_id", "lesion_id", "dx"]].copy()
    else:
        predictions_frame = pd.DataFrame({"row_index": np.arange(len(labels))})
    predictions_frame["true_index"] = labels
    predictions_frame["true_label"] = [label_names[int(index)] for index in labels]
    predictions_frame["predicted_index"] = predictions
    predictions_frame["predicted_label"] = [label_names[int(index)] for index in predictions]
    predictions_frame["correct"] = labels == predictions
    predictions_frame["predicted_probability"] = probabilities[np.arange(len(predictions)), predictions]
    predictions_frame["true_probability"] = probabilities[np.arange(len(labels)), labels]
    for index, label in enumerate(label_names):
        predictions_frame[f"probability_{label}"] = probabilities[:, index]
    predictions_frame.to_csv(output_dir / f"{strategy.name}_{split_name}_predictions.csv", index=False)


def resolve_split_path(splits_csv: str | None, features_path: Path) -> Path:
    return Path(splits_csv).expanduser().resolve() if splits_csv else features_path.parent / "splits.csv"


def load_split_rows(splits_csv: str | None, features_path: Path) -> dict[str, pd.DataFrame | None]:
    split_path = resolve_split_path(splits_csv, features_path)
    if not split_path.exists():
        return {"val": None, "test": None}
    split_frame = pd.read_csv(split_path)
    required = {"split", "image_id", "lesion_id", "dx"}
    if not required.issubset(split_frame.columns):
        return {"val": None, "test": None}
    return {
        split_name: split_frame[split_frame["split"] == split_name].reset_index(drop=True)
        for split_name in ("val", "test")
    }


if __name__ == "__main__":
    main()
