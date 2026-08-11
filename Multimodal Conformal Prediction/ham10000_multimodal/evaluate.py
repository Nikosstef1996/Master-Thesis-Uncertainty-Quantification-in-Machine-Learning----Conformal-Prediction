from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    HAM10000Dataset,
    load_ham10000_frame,
    make_image_transform,
    transform_metadata,
)
from .model import MultimodalClassifier
from .utils import classification_metrics, save_json, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HAM10000 classifier.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--data-dir", required=True, help="Directory containing HAM10000 images.")
    parser.add_argument("--metadata-csv", required=True, help="Path to HAM10000_metadata.csv.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    parser.add_argument(
        "--splits-csv",
        default=None,
        help="Optional splits.csv from training. Defaults to the checkpoint directory.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--predictions-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config: dict[str, Any] = checkpoint["config"]

    frame = load_ham10000_frame(args.data_dir, args.metadata_csv)
    frame = filter_split(frame, args.split, args.splits_csv, checkpoint_path.parent)

    metadata_spec = config["metadata_spec"]
    metadata_features = None if metadata_spec is None else transform_metadata(frame, metadata_spec)
    metadata_dim = int(config["metadata_dim"])
    target_names = config["target_names"]

    dataset = HAM10000Dataset(
        frame,
        metadata_features,
        make_image_transform(int(config["image_size"]), train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MultimodalClassifier(
        backbone=config["backbone"],
        num_classes=len(config["label_to_index"]),
        metadata_dim=metadata_dim,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    loss, metrics, predictions = evaluate(model, loader, criterion, device, target_names)

    result = {
        "split": args.split,
        "loss": loss,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "report": metrics["report"],
    }
    output_json = Path(args.output_json) if args.output_json else checkpoint_path.parent / f"{args.split}_metrics.json"
    save_json(result, output_json)

    if args.predictions_csv:
        predictions_frame = frame[["image_id", "dx"]].copy()
        predictions_frame["predicted_index"] = predictions["predicted_index"]
        predictions_frame["predicted_label"] = [
            target_names[index].split(": ", maxsplit=1)[0] for index in predictions["predicted_index"]
        ]
        predictions_frame.to_csv(args.predictions_csv, index=False)

    print(
        f"{args.split} | loss={loss:.4f} acc={metrics['accuracy']:.4f} "
        f"balanced_acc={metrics['balanced_accuracy']:.4f}"
    )


def filter_split(
    frame: pd.DataFrame,
    split: str,
    splits_csv: str | None,
    checkpoint_dir: Path,
) -> pd.DataFrame:
    if split == "all":
        return frame.reset_index(drop=True)

    split_path = Path(splits_csv).expanduser().resolve() if splits_csv else checkpoint_dir / "splits.csv"
    split_frame = pd.read_csv(split_path)
    selected_ids = set(split_frame.loc[split_frame["split"] == split, "image_id"].astype(str))
    filtered = frame[frame["image_id"].astype(str).isin(selected_ids)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"No rows found for split {split!r} using {split_path}")
    return filtered


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_names: list[str],
) -> tuple[float, dict[str, Any], dict[str, list[int]]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    total_count = 0
    for images, metadata, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images, metadata)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    metrics = classification_metrics(y_true, y_pred, target_names)
    return total_loss / max(total_count, 1), metrics, {"predicted_index": y_pred}


if __name__ == "__main__":
    main()
