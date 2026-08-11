import numpy as np
from sklearn.base import clone

class SplitConformalRegressor:
    """
    Split conformal regressor supporting:
    - residual
    - scaled_residual
    - cqr

    Model requirements by score type
    --------------------------------
    residual:
        point_model with fit(X, y), predict(X)

    scaled_residual:
        point_model with fit(X, y), predict(X)
        scale_model with fit(X, y), predict(X)

    cqr:
        lower_model with fit(X, y), predict(X)
        upper_model with fit(X, y), predict(X)
    """

    def __init__(
        self,
        score,
        alpha=0.1,
        point_model=None,
        scale_model=None,
        lower_model=None,
        upper_model=None,
    ):
        self.score = score
        self.alpha = alpha

        self.point_model = clone(point_model) if point_model is not None else None
        self.scale_model = clone(scale_model) if scale_model is not None else None
        self.lower_model = clone(lower_model) if lower_model is not None else None
        self.upper_model = clone(upper_model) if upper_model is not None else None

        self.q_hat_ = None
        self.calibration_scores_ = None
        self.is_fitted_ = False

    def _compute_qhat(self, scores):
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        return np.quantile(scores, q_level, method="higher")

    def fit(self, X_train, y_train, X_calib, y_calib):
        score_type = self.score.score_type

        if score_type == "residual":
            if self.point_model is None:
                raise ValueError("point_model is required for score_type='residual'.")

            self.point_model.fit(X_train, y_train)
            y_calib_pred = self.point_model.predict(X_calib)

            scores = self.score.compute(
                y_true=y_calib,
                y_pred=y_calib_pred,
            )

        elif score_type == "scaled_residual":
            if self.point_model is None:
                raise ValueError("point_model is required for score_type='scaled_residual'.")
            if self.scale_model is None:
                raise ValueError("scale_model is required for score_type='scaled_residual'.")

            self.point_model.fit(X_train, y_train)
            y_train_pred = self.point_model.predict(X_train)

            residuals_train = np.abs(y_train - y_train_pred)
            self.scale_model.fit(X_train, residuals_train)

            y_calib_pred = self.point_model.predict(X_calib)
            scale_calib = self.scale_model.predict(X_calib)
            scale_calib = np.maximum(scale_calib, 0.0)

            scores = self.score.compute(
                y_true=y_calib,
                y_pred=y_calib_pred,
                scale=scale_calib,
            )

        elif score_type == "cqr":
            if self.lower_model is None or self.upper_model is None:
                raise ValueError(
                    "lower_model and upper_model are required for score_type='cqr'."
                )

            self.lower_model.fit(X_train, y_train)
            self.upper_model.fit(X_train, y_train)

            lower_calib = self.lower_model.predict(X_calib) # Lower_model is from GradientBoostingRegressor with alpha=0.05
            upper_calib = self.upper_model.predict(X_calib) # Upper_model is from GradientBoostingRegressor with alpha=0.95
            lower_calib = np.minimum(lower_calib, upper_calib)
            upper_calib = np.maximum(lower_calib, upper_calib)

            scores = self.score.compute(
                y_true=y_calib,
                lower_pred=lower_calib,
                upper_pred=upper_calib,
            )

        else:
            raise ValueError(f"Unsupported score_type: {score_type}")

        scores = np.asarray(scores)
        if scores.ndim != 1:
            raise ValueError("Calibration scores must be a 1D array.")

        self.calibration_scores_ = scores
        self.q_hat_ = self._compute_qhat(scores)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")

        score_type = self.score.score_type

        if score_type in {"residual", "scaled_residual"}:
            return self.point_model.predict(X)

        if score_type == "cqr":
            lower_pred = self.lower_model.predict(X)
            upper_pred = self.upper_model.predict(X)
            return 0.5 * (lower_pred + upper_pred)

        raise ValueError(f"Unsupported score_type: {score_type}")

    def predict_interval(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")

        score_type = self.score.score_type

        if score_type == "residual":
            y_pred = self.point_model.predict(X)
            return self.score.interval(
                q_hat=self.q_hat_,
                y_pred=y_pred,
            )

        if score_type == "scaled_residual":
            y_pred = self.point_model.predict(X)
            scale = self.scale_model.predict(X)
            scale = np.maximum(scale, 0.0)

            return self.score.interval(
                q_hat=self.q_hat_,
                y_pred=y_pred,
                scale=scale,
            )

        if score_type == "cqr":
            lower_pred = self.lower_model.predict(X)
            upper_pred = self.upper_model.predict(X)

            lower_fixed = np.minimum(lower_pred, upper_pred)
            upper_fixed = np.maximum(lower_pred, upper_pred)

            return self.score.interval(
                q_hat=self.q_hat_,
                lower_pred=lower_fixed,
                upper_pred=upper_fixed,
            )

        raise ValueError(f"Unsupported score_type: {score_type}")