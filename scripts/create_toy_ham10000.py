from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
SEXES = ["male", "female", "unknown"]
LOCATIONS = ["back", "face", "lower extremity", "trunk"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny HAM10000-shaped toy dataset.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groups-per-class", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for label_idx, label in enumerate(LABELS):
        base = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
        base[..., label_idx % 3] = 80 + label_idx * 20
        for group_idx in range(args.groups_per_class):
            image_id = f"ISIC_toy_{label}_{group_idx:02d}"
            noise = rng.normal(0, 20, size=base.shape).astype(np.int16)
            image = np.clip(base.astype(np.int16) + noise + group_idx * 3, 0, 255).astype(
                np.uint8
            )
            Image.fromarray(image).save(image_dir / f"{image_id}.jpg", quality=90)
            rows.append(
                {
                    "lesion_id": f"lesion_{label}_{group_idx:02d}",
                    "image_id": image_id,
                    "dx": label,
                    "dx_type": "toy",
                    "age": 25 + label_idx * 5 + group_idx,
                    "sex": SEXES[(label_idx + group_idx) % len(SEXES)],
                    "localization": LOCATIONS[(label_idx + group_idx) % len(LOCATIONS)],
                }
            )

    metadata_path = root / "HAM10000_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {metadata_path}")


if __name__ == "__main__":
    main()
