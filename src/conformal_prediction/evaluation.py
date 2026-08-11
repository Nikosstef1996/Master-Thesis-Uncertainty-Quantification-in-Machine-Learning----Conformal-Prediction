from __future__ import annotations

from typing import Any

import numpy as np


def evaluate_prediction_sets(
    y_true: np.ndarray,
    prediction_sets: list[np.ndarray],
    classes_: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    classes_ = np.asarray(classes_)
    covered = np.asarray(
        [true_label in set(pred_set.tolist()) for true_label, pred_set in zip(y_true, prediction_sets)]
    )
    set_sizes = np.asarray([len(pred_set) for pred_set in prediction_sets], dtype=int)

    per_class = {}
    for class_label in classes_:
        mask = y_true == class_label
        if not mask.any():
            continue
        per_class[str(class_label)] = {
            "support": int(mask.sum()),
            "coverage": float(covered[mask].mean()),
            "average_set_size": float(set_sizes[mask].mean()),
        }

    return {
        "coverage": float(covered.mean()),
        "average_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "singleton_fraction": float((set_sizes == 1).mean()),
        "max_set_size": int(set_sizes.max(initial=0)),
        "covered": covered,
        "set_sizes": set_sizes,
        "per_class": per_class,
    }
