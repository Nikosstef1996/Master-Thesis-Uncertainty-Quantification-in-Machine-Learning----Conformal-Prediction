from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    HAM10000Dataset,
    LABEL_NAMES,
    fit_metadata_spec,
    load_ham10000_frame,
    make_grouped_splits,
    make_image_transform,
    save_splits,
    transform_metadata,
)
from .model import SUPPORTED_BACKBONES, build_image_encoder
from .utils import classification_metrics, save_json, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a classical multimodal HAM10000 classifier: "
            "CNN image features + metadata features -> Random Forest."
        )
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing HAM10000 images.")
    parser.add_argument("--metadata-csv", required=True, help="Path to HAM10000_metadata.csv.")
    parser.add_argument("--output-dir", default="runs/random_forest", help="Where to write outputs.")
    parser.add_argument("--backbone", default="resnet18", choices=SUPPORTED_BACKBONES)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use ImageNet weights.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional small debug subset.")

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
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"Using device for feature extraction: {device}")

    frame = load_ham10000_frame(args.data_dir, args.metadata_csv)
    if args.max_samples is not None:
        frame = frame.sample(n=min(args.max_samples, len(frame)), random_state=args.seed).reset_index(drop=True)

    splits = make_grouped_splits(frame, args.val_size, args.test_size, args.seed)
    save_splits(splits, output_dir)

    label_to_index = {label: idx for idx, label in enumerate(sorted(LABEL_NAMES))}
    label_names = sorted(LABEL_NAMES)
    target_names = [f"{label}: {LABEL_NAMES[label]}" for label in label_names]

    metadata_spec = fit_metadata_spec(splits.train)
    train_metadata = transform_metadata(splits.train, metadata_spec)
    val_metadata = transform_metadata(splits.val, metadata_spec)
    test_metadata = transform_metadata(splits.test, metadata_spec)

    image_encoder, image_dim = build_image_encoder(
        args.backbone,
        pretrained=not args.no_pretrained,
    )
    image_encoder.to(device)
    image_encoder.eval()

    train_features, train_labels = extract_fused_features(
        frame=splits.train,
        metadata_features=train_metadata,
        image_encoder=image_encoder,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        split_name="train",
    )
    val_features, val_labels = extract_fused_features(
        frame=splits.val,
        metadata_features=val_metadata,
        image_encoder=image_encoder,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        split_name="val",
    )
    test_features, test_labels = extract_fused_features(
        frame=splits.test,
        metadata_features=test_metadata,
        image_encoder=image_encoder,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        split_name="test",
    )

    feature_names = [
        *(f"image_feature_{index:04d}" for index in range(image_dim)),
        *metadata_spec["feature_names"],
    ]
    save_feature_arrays(
        output_dir,
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        feature_names,
    )

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

    print("Training RandomForestClassifier...")
    classifier.fit(train_features, train_labels)

    config = {
        **vars(args),
        "label_to_index": label_to_index,
        "target_names": target_names,
        "metadata_spec": metadata_spec,
        "image_feature_dim": image_dim,
        "metadata_dim": int(train_metadata.shape[1]),
        "fused_feature_dim": int(train_features.shape[1]),
        "model_type": "RandomForestClassifier",
        "fusion": "late feature concatenation",
    }
    save_json(config, output_dir / "config.json")
    joblib.dump(
        {
            "classifier": classifier,
            "config": config,
            "feature_names": feature_names,
        },
        output_dir / "random_forest.joblib",
    )

    save_feature_importances(classifier, feature_names, output_dir / "feature_importances.csv")
    save_split_outputs(classifier, splits.val, val_features, val_labels, label_names, target_names, output_dir, "val")
    save_split_outputs(classifier, splits.test, test_features, test_labels, label_names, target_names, output_dir, "test")


@torch.no_grad()
def extract_fused_features(
    frame: pd.DataFrame,
    metadata_features: np.ndarray,
    image_encoder: nn.Module,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = HAM10000Dataset(
        frame,
        metadata_features,
        make_image_transform(image_size, train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    image_feature_batches: list[np.ndarray] = []
    metadata_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    for images, metadata, labels in tqdm(loader, desc=f"features {split_name}", leave=False):
        images = images.to(device, non_blocking=True)
        image_features = image_encoder(images).detach().cpu().numpy().astype(np.float32)
        image_feature_batches.append(image_features)
        metadata_batches.append(metadata.numpy().astype(np.float32))
        label_batches.append(labels.numpy().astype(np.int64))

    image_features = np.concatenate(image_feature_batches, axis=0)
    metadata = np.concatenate(metadata_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    fused = np.concatenate([image_features, metadata], axis=1).astype(np.float32)
    return fused, labels


def save_feature_arrays(
    output_dir: Path,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    feature_names: list[str],
) -> None:
    np.savez_compressed(
        output_dir / "fused_features.npz",
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        feature_names=np.asarray(feature_names),
    )


def save_split_outputs(
    classifier: RandomForestClassifier,
    frame: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    target_names: list[str],
    output_dir: Path,
    split_name: str,
) -> None:
    predicted = classifier.predict(features).astype(int)
    probabilities = predict_proba_full(classifier, features, num_classes=len(label_names))
    metrics = classification_metrics(labels, predicted, target_names)

    save_json(
        {
            "split": split_name,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "report": metrics["report"],
        },
        output_dir / f"{split_name}_metrics.json",
    )
    pd.DataFrame(metrics["report"]).T.to_csv(output_dir / f"{split_name}_classification_report.csv")

    matrix = confusion_matrix(labels, predicted, labels=list(range(len(label_names))))
    pd.DataFrame(matrix, index=label_names, columns=label_names).to_csv(
        output_dir / f"{split_name}_confusion_matrix.csv"
    )

    predictions = frame[["image_id", "lesion_id", "dx"]].copy()
    predictions["true_index"] = labels
    predictions["predicted_index"] = predicted
    predictions["predicted_label"] = [label_names[index] for index in predicted]
    predictions["correct"] = predictions["true_index"] == predictions["predicted_index"]
    predictions["predicted_probability"] = probabilities[np.arange(len(predicted)), predicted]
    predictions["true_probability"] = probabilities[np.arange(len(labels)), labels]
    for index, label in enumerate(label_names):
        predictions[f"probability_{label}"] = probabilities[:, index]
    predictions.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    print(
        f"{split_name} | acc={metrics['accuracy']:.4f} "
        f"balanced_acc={metrics['balanced_accuracy']:.4f}"
    )


def predict_proba_full(
    classifier: RandomForestClassifier,
    features: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    probabilities = classifier.predict_proba(features)
    full = np.zeros((features.shape[0], num_classes), dtype=np.float32)
    for column_index, class_index in enumerate(classifier.classes_):
        full[:, int(class_index)] = probabilities[:, column_index]
    return full


def save_feature_importances(
    classifier: RandomForestClassifier,
    feature_names: list[str],
    path: Path,
) -> None:
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": classifier.feature_importances_,
        }
    )
    importances.sort_values("importance", ascending=False).to_csv(path, index=False)


if __name__ == "__main__":
    main()
