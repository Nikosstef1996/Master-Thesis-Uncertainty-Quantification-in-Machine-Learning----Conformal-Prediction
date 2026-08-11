from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone

from .scores import ClassificationScores


class SplitConformalClassifier:
    def __init__(
        self,
        model: Any,
        alpha: float = 0.1,
        score: ClassificationScores | None = None,
        prefit: bool = False,
    ) -> None:
        self.model = model if prefit else clone(model)
        self.alpha = alpha
        self.score = score or ClassificationScores("probability")

        self.q_hat_: float | None = None
        self.classes_: np.ndarray | None = None
        self.calibration_scores_: np.ndarray | None = None
        self.is_fitted_ = False

    def _conformal_quantile(self, scores: np.ndarray) -> float:
        n = len(scores)
        if n == 0:
            raise ValueError("Cannot calibrate conformal model with zero calibration samples.")
        q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        return float(np.quantile(scores, q_level, method="higher"))

    def _require_calibrated(self) -> None:
        if not self.is_fitted_ or self.q_hat_ is None or self.classes_ is None:
            raise RuntimeError("SplitConformalClassifier must be fitted or calibrated first.")

    def _true_label_positions(self, y: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model classes are not available.")
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        missing = sorted(set(y).difference(class_to_index))
        if missing:
            raise ValueError(f"Calibration labels not seen by model: {missing}")
        return np.asarray([class_to_index[label] for label in y], dtype=int)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> "SplitConformalClassifier":
        self.model.fit(X_train, y_train)
        return self.calibrate(X_calib, y_calib)

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray) -> "SplitConformalClassifier":
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Conformal classification requires a model with predict_proba.")
        if not hasattr(self.model, "classes_"):
            raise RuntimeError("The model must be fitted before calibration.")

        self.classes_ = np.asarray(self.model.classes_)
        y_calib = np.asarray(y_calib)
        true_class_indices = self._true_label_positions(y_calib)

        probs_calib = self.model.predict_proba(X_calib)
        scores = self.score.compute(probs_calib, true_class_indices)

        self.calibration_scores_ = scores
        self.q_hat_ = self._conformal_quantile(scores)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_set(self, X: np.ndarray) -> list[np.ndarray]:
        self._require_calibrated()
        probs = self.model.predict_proba(X)
        return self.score.build_set(probs, float(self.q_hat_), self.classes_)
