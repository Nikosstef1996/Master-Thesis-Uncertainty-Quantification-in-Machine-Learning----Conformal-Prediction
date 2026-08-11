import numpy as np
from sklearn.base import clone
from CP.scores import ClassificationScores


class SplitConformalClassifier:
    def __init__(self, model, alpha=0.1, score=None):
        self.model = clone(model)
        self.alpha = alpha
        self.score = score or ClassificationScores("probability")

        self.q_hat_ = None
        self.classes_ = None
        self.calibration_scores_ = None
        self.is_fitted_ = False

    def _conformal_quantile(self, scores):
        n = len(scores)
        q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        return np.quantile(scores, q_level, method="higher")

    def fit(self, X_train, y_train, X_calib, y_calib):
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_

        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        true_class_indices = np.array([class_to_index[y] for y in y_calib])

        probs_calib = self.model.predict_proba(X_calib)
        scores = self.score.compute(probs_calib, true_class_indices)

        self.calibration_scores_ = scores
        self.q_hat_ = self._conformal_quantile(scores)
        self.is_fitted_ = True

        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_set(self, X):
        probs = self.model.predict_proba(X)
        return self.score.build_set(probs, self.q_hat_, self.classes_)
