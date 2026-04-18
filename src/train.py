"""Training pipeline for the credit-risk model.

Lifts the logic from ``notebooks/03_modeling.ipynb`` into reusable functions
plus a ``main()`` entry point that reproduces the saved artifact at
``models/xgboost_calibrated.pkl``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .config import load_config, repo_root


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


def _ks_statistic(y_true, y_prob) -> float:
    """KS = max |CDF_default - CDF_non-default|.  Simple rank-based metric."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df = df.sort_values("p", ascending=False)
    n_pos = int(df["y"].sum())
    n_neg = int((df["y"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    cum_pos = (df["y"] == 1).cumsum() / n_pos
    cum_neg = (df["y"] == 0).cumsum() / n_neg
    return float((cum_pos - cum_neg).abs().max())


def load_processed_data(
    path: str | Path | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load the engineered dataset, label-encode categoricals, and split.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    cfg = load_config()
    if path is None:
        path = repo_root() / cfg["data"]["processed_dir"] / "train_engineered.csv"
    if test_size is None:
        test_size = cfg["data"]["test_size"]
    if random_state is None:
        random_state = cfg["data"]["random_seed"]

    df = pd.read_csv(path)

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> LogisticRegression:
    """Fit a class-balanced logistic-regression baseline."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> XGBClassifier:
    """Fit the gradient-boosted tree model.

    Default hyperparameters come from ``configs/config.yaml`` (model.params).
    ``scale_pos_weight`` is set to 11 to compensate for the ~8 % default rate.
    """
    cfg = load_config()
    defaults = dict(cfg["model"]["params"])
    defaults.update({
        "scale_pos_weight": 11,
        "eval_metric": "auc",
        "verbosity": 0,
    })
    if params:
        defaults.update(params)

    model = XGBClassifier(**defaults)
    model.fit(X_train, y_train)
    return model


def calibrate_model(
    base_model,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
) -> IsotonicCalibratedModel:
    """Fit an isotonic calibrator on ``(X_cal, y_cal)`` and wrap the model.

    Note: the reference notebook calibrates on the *test* set. That inflates
    apparent calibration quality but preserves the ranking. Pass a separate
    holdout here if a more defensible estimate is needed.
    """
    raw = base_model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, y_cal)
    return IsotonicCalibratedModel(base_model, iso)


def evaluate_model(model, X, y) -> dict[str, float]:
    """Compute AUC-ROC, PR-AUC, KS and Brier score on a held-out set.

    Returns a plain dict, rounded to 4 decimals, so it can be dropped into a
    pandas DataFrame row without further processing.
    """
    prob = model.predict_proba(X)[:, 1]
    return {
        "AUC-ROC": round(float(roc_auc_score(y, prob)), 4),
        "PR-AUC": round(float(average_precision_score(y, prob)), 4),
        "KS": round(_ks_statistic(y, prob), 4),
        "Brier": round(float(brier_score_loss(y, prob)), 4),
    }


def main() -> dict[str, dict[str, float]]:
    """Run the full training pipeline end-to-end.

    Trains LR + XGBoost, calibrates XGBoost with isotonic regression,
    evaluates all three, prints a comparison table, and saves the
    calibrated model to ``models/xgboost_calibrated.pkl``.
    """
    cfg = load_config()

    X_train, X_test, y_train, y_test, _ = load_processed_data()

    lr = train_logistic_regression(X_train, y_train, random_state=cfg["data"]["random_seed"])
    xgb = train_xgboost(X_train, y_train)
    xgb_cal = calibrate_model(xgb, X_test, y_test)

    metrics = {
        "Logistic Regression": evaluate_model(lr, X_test, y_test),
        "XGBoost (raw)": evaluate_model(xgb, X_test, y_test),
        "XGBoost (calibrated)": evaluate_model(xgb_cal, X_test, y_test),
    }

    comparison = pd.DataFrame(metrics).T
    print("=== Model Comparison (Test Set) ===")
    print(comparison.to_string())

    model_dir = repo_root() / cfg["output"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost_calibrated.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(xgb_cal, f)
    print(f"\nCalibrated model saved to {model_path}")

    return metrics


if __name__ == "__main__":
    main()
