from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from .data import make_image_transform, transform_metadata
from .model import MultimodalClassifier
from .utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction for one skin-lesion image.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--image", required=True, help="Path to a lesion image.")
    parser.add_argument("--age", type=float, default=None)
    parser.add_argument("--sex", default="unknown")
    parser.add_argument("--localization", default="unknown")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    checkpoint = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location=device)
    config = checkpoint["config"]

    model = MultimodalClassifier(
        backbone=config["backbone"],
        num_classes=len(config["label_to_index"]),
        metadata_dim=int(config["metadata_dim"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    transform = make_image_transform(int(config["image_size"]), train=False)
    image = Image.open(Path(args.image).expanduser().resolve()).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    metadata_spec = config["metadata_spec"]
    if metadata_spec is None:
        metadata_tensor = torch.empty((1, 0), dtype=torch.float32, device=device)
    else:
        row = pd.DataFrame(
            [
                {
                    "age": args.age,
                    "sex": args.sex,
                    "localization": args.localization,
                }
            ]
        )
        metadata_tensor = torch.tensor(
            transform_metadata(row, metadata_spec),
            dtype=torch.float32,
            device=device,
        )

    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor, metadata_tensor), dim=1)[0]

    index_to_label = {value: key for key, value in config["label_to_index"].items()}
    target_names = {
        item.split(": ", maxsplit=1)[0]: item.split(": ", maxsplit=1)[1]
        for item in config["target_names"]
    }
    top_k = min(args.top_k, probabilities.numel())
    values, indices = torch.topk(probabilities, k=top_k)
    for probability, index in zip(values.tolist(), indices.tolist(), strict=True):
        label = index_to_label[index]
        print(f"{label:5s} {probability:.4f} {target_names[label]}")


if __name__ == "__main__":
    main()
