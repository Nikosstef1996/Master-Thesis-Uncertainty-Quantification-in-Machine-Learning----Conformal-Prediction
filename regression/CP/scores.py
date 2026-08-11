import numpy as np
from sklearn.base import clone


class RegressionScores:
    """
    Score class for conformal regression.

    Supported score types
    ---------------------
    - "residual"
    - "scaled_residual"
    - "cqr"
    """

    def __init__(self, score_type="residual", eps=1e-8):
        valid_scores = {"residual", "scaled_residual", "cqr"}
        if score_type not in valid_scores:
            raise ValueError(
                f"Unknown score_type '{score_type}'. "
                f"Valid options are: {sorted(valid_scores)}"
            )

        self.score_type = score_type
        self.eps = eps

    def compute(
        self,
        y_true,
        y_pred=None,
        scale=None,
        lower_pred=None,
        upper_pred=None,
    ):
        """
        Compute calibration scores.
        """
        y_true = np.asarray(y_true)

        if self.score_type == "residual":
            if y_pred is None:
                raise ValueError("y_pred is required for score_type='residual'.")
            y_pred = np.asarray(y_pred)
            return np.abs(y_true - y_pred)

        if self.score_type == "scaled_residual":
            if y_pred is None:
                raise ValueError("y_pred is required for score_type='scaled_residual'.")
            if scale is None:
                raise ValueError("scale is required for score_type='scaled_residual'.")

            y_pred = np.asarray(y_pred)
            scale = np.asarray(scale)

            if np.any(scale < 0):
                raise ValueError("scale must be non-negative.")

            return np.abs(y_true - y_pred) / (scale + self.eps)

        if self.score_type == "cqr":
            if lower_pred is None or upper_pred is None:
                raise ValueError(
                    "lower_pred and upper_pred are required for score_type='cqr'."
                )

            lower_pred = np.asarray(lower_pred)
            upper_pred = np.asarray(upper_pred)

            if np.any(lower_pred > upper_pred):
                raise ValueError("lower_pred must be <= upper_pred elementwise.")

            return np.maximum(lower_pred - y_true, y_true - upper_pred)

        raise RuntimeError("Unreachable branch.")

    def interval(
        self,
        q_hat,
        y_pred=None,
        scale=None,
        lower_pred=None,
        upper_pred=None,
    ):
        """
        Build conformal prediction intervals from q_hat.
        """
        if q_hat is None:
            raise ValueError("q_hat must be provided.")

        if self.score_type == "residual":
            if y_pred is None:
                raise ValueError("y_pred is required for score_type='residual'.")
            y_pred = np.asarray(y_pred)
            return y_pred - q_hat, y_pred + q_hat

        if self.score_type == "scaled_residual":
            if y_pred is None:
                raise ValueError("y_pred is required for score_type='scaled_residual'.")
            if scale is None:
                raise ValueError("scale is required for score_type='scaled_residual'.")

            y_pred = np.asarray(y_pred)
            scale = np.asarray(scale)

            if np.any(scale < 0):
                raise ValueError("scale must be non-negative.")

            radius = q_hat * scale
            return y_pred - radius, y_pred + radius

        if self.score_type == "cqr":
            if lower_pred is None or upper_pred is None:
                raise ValueError(
                    "lower_pred and upper_pred are required for score_type='cqr'."
                )

            lower_pred = np.asarray(lower_pred)
            upper_pred = np.asarray(upper_pred)

            if np.any(lower_pred > upper_pred):
                raise ValueError("lower_pred must be <= upper_pred elementwise.")

            return lower_pred - q_hat, upper_pred + q_hat

        raise RuntimeError("Unreachable branch.")


