"""Model wrapper classes shared across training, scoring, and the dashboard.

Kept import-light (numpy + sklearn only) so that pickle.load can resolve the
class without dragging in mlflow, xgboost, matplotlib, etc.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibratedModel:
    """Wrap a fitted binary classifier with an isotonic-regression calibrator.

    Exposes the sklearn ``predict_proba`` / ``predict`` interface so downstream
    code (risk policy, SHAP explainers) can treat it as a drop-in model. The
    raw tree model is kept accessible as ``base_model`` because SHAP runs on
    the tree, not the post-hoc calibrator.
    """

    def __init__(self, base_model, iso_reg: IsotonicRegression):
        self.base_model = base_model
        self.iso_reg = iso_reg

    def predict_proba(self, X) -> np.ndarray:
        """Return calibrated class probabilities, shape ``(n, 2)``."""
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.iso_reg.predict(raw)
        return np.column_stack([1 - cal, cal])

    def predict(self, X) -> np.ndarray:
        """Hard predictions at ``p >= 0.5`` on calibrated probability."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


__all__ = ["IsotonicCalibratedModel"]
