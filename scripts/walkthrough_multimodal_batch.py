from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from ham10000_multimodal.data import (
    HAM10000Dataset,
    LABEL_NAMES,
    fit_metadata_spec,
    load_ham10000_frame,
    make_grouped_splits,
    make_image_transform,
    transform_metadata,
)
from ham10000_multimodal.model import MultimodalClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show how one HAM10000 multimodal batch flows through the PyTorch model."
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--metadata-csv", default="HAM10000_metadata.csv")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = load_ham10000_frame(args.data_dir, args.metadata_csv)
    splits = make_grouped_splits(frame, val_size=0.15, test_size=0.15, seed=args.seed)

    metadata_spec = fit_metadata_spec(splits.train)
    train_metadata = transform_metadata(splits.train, metadata_spec)

    dataset = HAM10000Dataset(
        frame=splits.train,
        metadata_features=train_metadata,
        transform=make_image_transform(args.image_size, train=True),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    images, metadata, labels = next(iter(loader))
    model = MultimodalClassifier(
        backbone=args.backbone,
        num_classes=len(LABEL_NAMES),
        metadata_dim=metadata.shape[1],
        pretrained=False,
    )
    logits = model(images, metadata)

    print("Dataset rows:", len(frame))
    print("Train rows:", len(splits.train))
    print("Metadata feature names:")
    for index, name in enumerate(metadata_spec["feature_names"]):
        print(f"  {index:02d}: {name}")
    print()
    print("Batch image tensor:", tuple(images.shape))
    print("Batch metadata tensor:", tuple(metadata.shape))
    print("Batch labels tensor:", tuple(labels.shape))
    print("Model logits tensor:", tuple(logits.shape))
    print()
    print("Meaning:")
    print(f"  images:   {args.batch_size} RGB images, resized to {args.image_size}x{args.image_size}")
    print(f"  metadata: {metadata.shape[1]} tabular features per image")
    print(f"  logits:   {len(LABEL_NAMES)} class scores per image")


if __name__ == "__main__":
    main()
