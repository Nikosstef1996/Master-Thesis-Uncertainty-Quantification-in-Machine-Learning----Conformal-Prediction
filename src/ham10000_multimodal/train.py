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
    LABEL_NAMES,
    compute_class_weights,
    fit_metadata_spec,
    load_ham10000_frame,
    make_grouped_splits,
    make_image_transform,
    save_splits,
    transform_metadata,
)
from .model import SUPPORTED_BACKBONES, MultimodalClassifier
from .utils import AverageMeter, classification_metrics, save_json, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multimodal HAM10000 classifier.")
    parser.add_argument("--data-dir", required=True, help="Directory containing HAM10000 images.")
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="Path to HAM10000_metadata.csv.",
    )
    parser.add_argument("--output-dir", default="runs/ham10000", help="Where to write checkpoints.")
    parser.add_argument("--backbone", default="resnet18", choices=SUPPORTED_BACKBONES)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights.")
    parser.add_argument("--no-metadata", action="store_true", help="Train an image-only classifier.")
    parser.add_argument("--freeze-image-encoder", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"Using device: {device}")

    frame = load_ham10000_frame(args.data_dir, args.metadata_csv)
    splits = make_grouped_splits(frame, args.val_size, args.test_size, args.seed)
    save_splits(splits, output_dir)

    label_to_index = {label: idx for idx, label in enumerate(sorted(LABEL_NAMES))}
    target_names = [f"{label}: {LABEL_NAMES[label]}" for label in sorted(LABEL_NAMES)]

    metadata_spec: dict[str, Any] | None
    if args.no_metadata:
        metadata_spec = None
        train_metadata = val_metadata = test_metadata = None
        metadata_dim = 0
    else:
        metadata_spec = fit_metadata_spec(splits.train)
        train_metadata = transform_metadata(splits.train, metadata_spec)
        val_metadata = transform_metadata(splits.val, metadata_spec)
        test_metadata = transform_metadata(splits.test, metadata_spec)
        metadata_dim = train_metadata.shape[1]

    train_dataset = HAM10000Dataset(
        splits.train,
        train_metadata,
        make_image_transform(args.image_size, train=True),
    )
    val_dataset = HAM10000Dataset(
        splits.val,
        val_metadata,
        make_image_transform(args.image_size, train=False),
    )
    test_dataset = HAM10000Dataset(
        splits.test,
        test_metadata,
        make_image_transform(args.image_size, train=False),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = MultimodalClassifier(
        backbone=args.backbone,
        num_classes=len(label_to_index),
        metadata_dim=metadata_dim,
        pretrained=args.pretrained,
    )
    if args.freeze_image_encoder:
        model.freeze_image_encoder()
    model.to(device)

    class_weights = compute_class_weights(
        splits.train["label_index"].to_numpy(),
        num_classes=len(label_to_index),
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    training_config = {
        **vars(args),
        "label_to_index": label_to_index,
        "target_names": target_names,
        "metadata_spec": metadata_spec,
        "metadata_dim": metadata_dim,
    }
    save_json(training_config, output_dir / "config.json")

    best_balanced_accuracy = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )
        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            target_names=target_names,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
        )

        if val_metrics["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = val_metrics["balanced_accuracy"]
            save_checkpoint(output_dir / "best.pt", model, training_config, epoch, val_metrics)

        save_checkpoint(output_dir / "last.pt", model, training_config, epoch, val_metrics)

    test_loss, test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        target_names=target_names,
    )
    save_json(
        {
            "test_loss": test_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_report": test_metrics["report"],
        },
        output_dir / "test_metrics.json",
    )
    print(
        f"Test | loss={test_loss:.4f} acc={test_metrics['accuracy']:.4f} "
        f"balanced_acc={test_metrics['balanced_accuracy']:.4f}"
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    losses = AverageMeter()
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for images, metadata, labels in progress:
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images, metadata)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(float(loss.detach().cpu()), images.size(0))
        progress.set_postfix(loss=f"{losses.average:.4f}")
    return losses.average


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_names: list[str],
) -> tuple[float, dict[str, Any]]:
    model.eval()
    losses = AverageMeter()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, metadata, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, metadata)
        loss = criterion(logits, labels)

        losses.update(float(loss.detach().cpu()), images.size(0))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    return losses.average, classification_metrics(y_true, y_pred, target_names)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


if __name__ == "__main__":
    main()
