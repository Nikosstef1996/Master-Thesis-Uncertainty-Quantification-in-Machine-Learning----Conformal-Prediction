from __future__ import annotations

import numpy as np


class ClassificationScores:
    def __init__(self, score_type: str = "probability") -> None:
        self.score_type = score_type

    def compute(self, probs: np.ndarray, true_class_indices: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs)
        true_class_indices = np.asarray(true_class_indices)
        n = probs.shape[0]

        if self.score_type == "probability":
            return 1.0 - probs[np.arange(n), true_class_indices]

        if self.score_type == "cumulative":
            scores = []
            for p, true_idx in zip(probs, true_class_indices):
                order = np.argsort(-p)
                sorted_probs = p[order]
                cum_probs = np.cumsum(sorted_probs)
                rank_pos = np.where(order == true_idx)[0][0]
                scores.append(cum_probs[rank_pos])
            return np.asarray(scores)

        if self.score_type == "high_probability":
            return -probs[np.arange(n), true_class_indices]

        raise ValueError(f"Unknown score_type: {self.score_type}")

    def build_set(
        self,
        probs: np.ndarray,
        q_hat: float,
        classes_: np.ndarray,
    ) -> list[np.ndarray]:
        probs = np.asarray(probs)
        classes_ = np.asarray(classes_)
        prediction_sets = []

        if self.score_type == "probability":
            for p in probs:
                mask = (1.0 - p) <= q_hat
                labels = classes_[mask]
                if len(labels) == 0:
                    labels = classes_[np.argmax(p) : np.argmax(p) + 1]
                prediction_sets.append(labels)
            return prediction_sets

        if self.score_type == "cumulative":
            for p in probs:
                order = np.argsort(-p)
                sorted_probs = p[order]
                cum_probs = np.cumsum(sorted_probs)
                included = order[cum_probs <= q_hat]
                if len(included) < len(order):
                    included = order[: len(included) + 1]
                if len(included) == 0:
                    included = order[:1]
                prediction_sets.append(classes_[included])
            return prediction_sets

        if self.score_type == "high_probability":
            for p in probs:
                mask = p >= -q_hat
                labels = classes_[mask]
                if len(labels) == 0:
                    labels = classes_[np.argmax(p) : np.argmax(p) + 1]
                prediction_sets.append(labels)
            return prediction_sets

        raise ValueError(f"Unknown score_type: {self.score_type}")
