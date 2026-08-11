from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import LABEL_NAMES, compute_class_weights
from .fusion_comparison import get_feature_groups, load_feature_arrays, load_split_rows
from .utils import classification_metrics, save_json, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an attention-based HAM10000 fusion model from saved "
            "ResNet image features and metadata features."
        )
    )
    parser.add_argument("--features-npz", required=True, help="Path to fused_features.npz.")
    parser.add_argument("--output-dir", default=None, help="Where to save outputs.")
    parser.add_argument("--splits-csv", default=None, help="Optional splits.csv for prediction files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Training device.",
    )
    return parser.parse_args()


class AttentionFusionClassifier(nn.Module):
    def __init__(
        self,
        image_dim: int,
        metadata_dim: int,
        hidden_dim: int,
        num_classes: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden-dim must be divisible by attention-heads.")

        self.image_encoder = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(metadata_dim),
            nn.Linear(metadata_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.modality_embedding = nn.Parameter(torch.zeros(2, hidden_dim))
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_features: torch.Tensor, metadata_features: torch.Tensor) -> torch.Tensor:
        image_token = self.image_encoder(image_features)
        metadata_token = self.metadata_encoder(metadata_features)
        tokens = torch.stack([image_token, metadata_token], dim=1)
        tokens = tokens + self.modality_embedding.unsqueeze(0)

        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.attention_norm(tokens + self.attention_dropout(attended))
        updated = self.feed_forward(tokens)
        tokens = self.feed_forward_norm(tokens + self.feed_forward_dropout(updated))
        return self.classifier(tokens)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    features_path = Path(args.features_npz).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else features_path.parent / "attention_fusion"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_feature_arrays(features_path)
    feature_names = [str(name) for name in arrays["feature_names"]]
    feature_groups = get_feature_groups(feature_names)
    features = split_features(arrays, feature_groups)

    label_names = sorted(LABEL_NAMES)
    target_names = [f"{label}: {LABEL_NAMES[label]}" for label in label_names]
    num_classes = len(label_names)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = AttentionFusionClassifier(
        image_dim=features["train"]["image"].shape[1],
        metadata_dim=features["train"]["metadata"].shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
    ).to(device)

    train_loader = make_loader(features["train"], arrays["train_labels"], args.batch_size, shuffle=True)
    val_loader = make_loader(features["val"], arrays["val_labels"], args.batch_size, shuffle=False)
    test_loader = make_loader(features["test"], arrays["test_labels"], args.batch_size, shuffle=False)

    class_weights = compute_class_weights(arrays["train_labels"], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(3, args.patience // 3),
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        target_names=target_names,
        epochs=args.epochs,
        patience=args.patience,
        output_dir=output_dir,
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    checkpoint = torch.load(output_dir / "best_attention_fusion.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    split_rows = load_split_rows(args.splits_csv, features_path)
    val_loss, val_metrics, val_predictions = evaluate(model, val_loader, criterion, device, target_names)
    test_loss, test_metrics, test_predictions = evaluate(model, test_loader, criterion, device, target_names)

    save_split_outputs(
        output_dir,
        "val",
        arrays["val_labels"],
        val_predictions,
        val_loss,
        val_metrics,
        label_names,
        target_names,
        split_rows.get("val"),
    )
    save_split_outputs(
        output_dir,
        "test",
        arrays["test_labels"],
        test_predictions,
        test_loss,
        test_metrics,
        label_names,
        target_names,
        split_rows.get("test"),
    )

    config = {
        **vars(args),
        "features_npz": str(features_path),
        "image_feature_dim": len(feature_groups["image"]),
        "metadata_dim": len(feature_groups["metadata"]),
        "metadata_feature_names": [feature_names[index] for index in feature_groups["metadata"]],
        "fusion": "attention over image and metadata modality tokens",
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_balanced_accuracy": float(checkpoint["val_balanced_accuracy"]),
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
    }
    save_json(config, output_dir / "config.json")

    print("\n========== HAM10000 ATTENTION FUSION ==========")
    print(
        f"val  | loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} "
        f"balanced_acc={val_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"test | loss={test_loss:.4f} acc={test_metrics['accuracy']:.4f} "
        f"balanced_acc={test_metrics['balanced_accuracy']:.4f}"
    )
    print(f"Saved attention-fusion outputs to: {output_dir}")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return select_device()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available.")
    return device


def split_features(
    arrays: dict[str, np.ndarray],
    feature_groups: dict[str, list[int]],
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("train", "val", "test"):
        all_features = arrays[f"{split_name}_features"]
        result[split_name] = {
            "image": all_features[:, feature_groups["image"]].astype(np.float32),
            "metadata": all_features[:, feature_groups["metadata"]].astype(np.float32),
        }
    return result


def make_loader(
    split_features: dict[str, np.ndarray],
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(split_features["image"], dtype=torch.float32),
        torch.tensor(split_features["metadata"], dtype=torch.float32),
        torch.tensor(labels.astype(np.int64), dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
    target_names: list[str],
    epochs: int,
    patience: int,
    output_dir: Path,
) -> list[dict[str, float]]:
    history = []
    best_val_balanced = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, target_names)
        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device, target_names)
        scheduler.step(val_metrics["balanced_accuracy"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
        )

        if val_metrics["balanced_accuracy"] > best_val_balanced:
            best_val_balanced = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_balanced_accuracy": best_val_balanced,
                },
                output_dir / "best_attention_fusion.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    return history


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_names: list[str],
) -> tuple[float, dict[str, Any]]:
    model.train()
    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for image_features, metadata_features, labels in tqdm(loader, desc="train", leave=False):
        image_features = image_features.to(device)
        metadata_features = metadata_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image_features, metadata_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    return total_loss / max(total_count, 1), classification_metrics(y_true, y_pred, target_names)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_names: list[str],
) -> tuple[float, dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    probability_batches: list[np.ndarray] = []

    for image_features, metadata_features, labels in tqdm(loader, desc="eval", leave=False):
        image_features = image_features.to(device)
        metadata_features = metadata_features.to(device)
        labels = labels.to(device)

        logits = model(image_features, metadata_features)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        probability_batches.append(probabilities.detach().cpu().numpy().astype(np.float32))

    probabilities = np.concatenate(probability_batches, axis=0)
    predictions = {
        "predicted_index": np.asarray(y_pred, dtype=np.int64),
        "probabilities": probabilities,
    }
    return total_loss / max(total_count, 1), classification_metrics(y_true, y_pred, target_names), predictions


def save_split_outputs(
    output_dir: Path,
    split_name: str,
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    loss: float,
    metrics: dict[str, Any],
    label_names: list[str],
    target_names: list[str],
    rows: pd.DataFrame | None,
) -> None:
    predicted = predictions["predicted_index"]
    probabilities = predictions["probabilities"]
    save_json(
        {
            "split": split_name,
            "loss": loss,
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

    if rows is not None and len(rows) == len(labels):
        frame = rows[["image_id", "lesion_id", "dx"]].copy()
    else:
        frame = pd.DataFrame({"row_index": np.arange(len(labels))})
    frame["true_index"] = labels
    frame["true_label"] = [label_names[int(index)] for index in labels]
    frame["predicted_index"] = predicted
    frame["predicted_label"] = [label_names[int(index)] for index in predicted]
    frame["correct"] = labels == predicted
    frame["predicted_probability"] = probabilities[np.arange(len(predicted)), predicted]
    frame["true_probability"] = probabilities[np.arange(len(labels)), labels]
    for index, label in enumerate(label_names):
        frame[f"probability_{label}"] = probabilities[:, index]
    frame.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)


if __name__ == "__main__":
    main()
