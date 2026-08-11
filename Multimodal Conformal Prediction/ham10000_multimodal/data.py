from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_NAMES = {
    "akiec": "Actinic keratosis / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}

CATEGORICAL_COLUMNS = ("sex", "localization")
NUMERIC_COLUMNS = ("age",)
UNKNOWN_CATEGORY = "unknown"


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class HAM10000Dataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        metadata_features: np.ndarray | None,
        transform: transforms.Compose,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.metadata_features = metadata_features
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image_tensor = self.transform(image)

        if self.metadata_features is None:
            metadata_tensor = torch.empty(0, dtype=torch.float32)
        else:
            metadata_tensor = torch.tensor(self.metadata_features[index], dtype=torch.float32)

        label = torch.tensor(int(row["label_index"]), dtype=torch.long)
        return image_tensor, metadata_tensor, label


def load_ham10000_frame(data_dir: str | Path, metadata_csv: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir).expanduser().resolve()
    metadata_csv = Path(metadata_csv).expanduser().resolve()
    frame = pd.read_csv(metadata_csv)

    required = {"image_id", "dx"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    frame["dx"] = frame["dx"].astype(str)

    image_index = build_image_index(data_dir)
    frame["image_path"] = frame["image_id"].map(image_index)
    missing_images = frame[frame["image_path"].isna()]["image_id"].tolist()
    if missing_images:
        preview = ", ".join(missing_images[:10])
        raise FileNotFoundError(
            f"Could not find {len(missing_images)} images under {data_dir}. "
            f"First missing image IDs: {preview}"
        )

    labels = sorted(frame["dx"].unique())
    unknown_labels = sorted(set(labels).difference(LABEL_NAMES))
    if unknown_labels:
        raise ValueError(
            f"Unexpected labels in dx column: {unknown_labels}. "
            f"Expected labels like {sorted(LABEL_NAMES)}."
        )

    label_to_index = {label: idx for idx, label in enumerate(sorted(LABEL_NAMES))}
    frame["label_index"] = frame["dx"].map(label_to_index).astype(int)
    if "lesion_id" not in frame.columns:
        frame["lesion_id"] = frame["image_id"]
    frame["lesion_id"] = frame["lesion_id"].fillna(frame["image_id"]).astype(str)
    return frame


def build_image_index(data_dir: Path) -> dict[str, str]:
    suffixes = {".jpg", ".jpeg", ".png"}
    image_paths = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    image_index: dict[str, str] = {}
    for path in image_paths:
        image_index.setdefault(path.stem, str(path))
    if not image_index:
        raise FileNotFoundError(f"No image files found under {data_dir}")
    return image_index


def make_grouped_splits(
    frame: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> SplitFrames:
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1")

    groups = frame["lesion_id"].astype(str).to_numpy()
    train_idx, holdout_idx = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=val_size + test_size,
            random_state=seed,
        ).split(frame, groups=groups)
    )

    holdout = frame.iloc[holdout_idx].reset_index(drop=True)
    holdout_groups = holdout["lesion_id"].astype(str).to_numpy()
    relative_test_size = test_size / (val_size + test_size)
    val_idx, test_idx = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=relative_test_size,
            random_state=seed + 1,
        ).split(holdout, groups=holdout_groups)
    )

    return SplitFrames(
        train=frame.iloc[train_idx].reset_index(drop=True),
        val=holdout.iloc[val_idx].reset_index(drop=True),
        test=holdout.iloc[test_idx].reset_index(drop=True),
    )


def make_image_transform(image_size: int, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def fit_metadata_spec(frame: pd.DataFrame) -> dict[str, Any]:
    age = _numeric_series(frame, "age")
    age_mean = float(age.mean()) if age.notna().any() else 0.0
    age_std = float(age.std()) if age.notna().sum() > 1 else 1.0
    if age_std == 0.0 or np.isnan(age_std):
        age_std = 1.0

    categorical_values: dict[str, list[str]] = {}
    for column in CATEGORICAL_COLUMNS:
        if column in frame.columns:
            values = _clean_category_series(frame[column])
        else:
            values = pd.Series([UNKNOWN_CATEGORY])
        unique_values = sorted(set(values.tolist()) | {UNKNOWN_CATEGORY})
        categorical_values[column] = unique_values

    feature_names = ["age_scaled"]
    for column, values in categorical_values.items():
        feature_names.extend(f"{column}={value}" for value in values)

    return {
        "age_mean": age_mean,
        "age_std": age_std,
        "categorical_values": categorical_values,
        "feature_names": feature_names,
    }


def transform_metadata(frame: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    features = np.zeros((len(frame), len(spec["feature_names"])), dtype=np.float32)

    age = _numeric_series(frame, "age").fillna(spec["age_mean"])
    features[:, 0] = ((age.to_numpy(dtype=np.float32) - spec["age_mean"]) / spec["age_std"]).astype(
        np.float32
    )

    offset = 1
    for column in CATEGORICAL_COLUMNS:
        values = spec["categorical_values"][column]
        value_to_index = {value: idx for idx, value in enumerate(values)}
        if column in frame.columns:
            cleaned = _clean_category_series(frame[column])
        else:
            cleaned = pd.Series([UNKNOWN_CATEGORY] * len(frame))

        for row_idx, value in enumerate(cleaned):
            normalized = value if value in value_to_index else UNKNOWN_CATEGORY
            features[row_idx, offset + value_to_index[normalized]] = 1.0
        offset += len(values)

    return features


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=num_classes)
    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = total / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def save_splits(split_frames: SplitFrames, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    rows = []
    for split_name, split_frame in (
        ("train", split_frames.train),
        ("val", split_frames.val),
        ("test", split_frames.test),
    ):
        split_rows = split_frame[["image_id", "lesion_id", "dx"]].copy()
        split_rows["split"] = split_name
        rows.append(split_rows)
    pd.concat(rows, ignore_index=True).to_csv(output_dir / "splits.csv", index=False)


def _clean_category_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna(UNKNOWN_CATEGORY).astype(str).str.strip().str.lower()
    cleaned = cleaned.replace({"": UNKNOWN_CATEGORY, "nan": UNKNOWN_CATEGORY, "none": UNKNOWN_CATEGORY})
    return cleaned


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([np.nan] * len(frame))
    return pd.to_numeric(frame[column], errors="coerce")
