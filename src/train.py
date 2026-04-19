"""Training pipeline for the credit-risk model.

Lifts the logic from ``notebooks/03_modeling.ipynb`` into reusable functions
plus a ``main()`` entry point that runs two tracked MLflow experiments
(Logistic Regression, then XGBoost + isotonic calibration).
"""
from __future__ import annotations

import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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
# Re-exported so that pickles serialised with ``src.train.IsotonicCalibratedModel``
# (the original module path) still resolve after the class moved to ``src.models``.
from .models import IsotonicCalibratedModel  # noqa: F401


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


def _brier_skill_score(y_true, y_prob) -> float:
    """Brier Skill Score: 1 - Brier / Brier_base. Positive = beats baseline."""
    base_rate = float(np.mean(y_true))
    brier_base = base_rate * (1 - base_rate)
    if brier_base == 0:
        return 0.0
    return 1 - float(brier_score_loss(y_true, y_prob)) / brier_base


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
    """Compute AUC-ROC, PR-AUC, KS, Brier and Brier skill score on a holdout.

    Returns a plain dict, rounded to 4 decimals, so it can be dropped into a
    pandas DataFrame row without further processing.
    """
    prob = model.predict_proba(X)[:, 1]
    return {
        "AUC-ROC": round(float(roc_auc_score(y, prob)), 4),
        "PR-AUC": round(float(average_precision_score(y, prob)), 4),
        "KS": round(_ks_statistic(y, prob), 4),
        "Brier": round(float(brier_score_loss(y, prob)), 4),
        "Brier_Skill": round(_brier_skill_score(y, prob), 4),
    }


# --- MLflow helpers -----------------------------------------------------------


def _save_calibration_plot(
    y_true: np.ndarray,
    probs: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Write a reliability diagram + score histogram to ``out_path``."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    frac_pos, mean_pred = calibration_curve(
        y_true, probs, n_bins=15, strategy="quantile"
    )
    axes[0].plot(mean_pred, frac_pos, marker="o", lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Fraction of positives")
    axes[0].set_title("Reliability Diagram")
    axes[0].legend()

    axes[1].hist(probs, bins=50, density=True, alpha=0.7)
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Score Distribution")

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_feature_importance_plot(
    model, feature_names: list[str], out_path: Path, top_n: int = 20
) -> None:
    """Write a horizontal bar chart of the model's top-``top_n`` features.

    Supports either tree models (``feature_importances_``) or linear models
    (``coef_``, plotted as absolute value).
    """
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        xlabel = "Feature importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_)).ravel()
        xlabel = "|Coefficient|"
    else:
        return  # nothing sensible to plot

    series = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(series) + 1))
    series[::-1].plot.barh(ax=ax, edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top {top_n} Features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _log_portfolio_metrics(model, X_test, y_test, t_low: float, t_high: float):
    """Log conservative-scenario portfolio metrics to the active MLflow run."""
    import mlflow  # lazy: keeps mlflow optional for non-training callers
    # Lazy import to avoid a policy -> train cycle at import time.
    from .policy import simulate_portfolio

    probs = model.predict_proba(X_test)[:, 1]
    sim = simulate_portfolio(
        probs, y_test.values, X_test["AMT_CREDIT"].values,
        t_low=t_low, t_high=t_high,
    )
    cons = sim.loc["Conservative (LOW only)"]
    mlflow.log_metric("approval_rate_conservative", float(cons["approval_rate"]))
    mlflow.log_metric("portfolio_default_rate_conservative", float(cons["portfolio_default_rate"]))
    mlflow.log_metric("expected_loss_conservative", float(cons["expected_loss"]))


def _run_lr_experiment(
    X_train, X_test, y_train, y_test, feature_names: list[str],
    cfg: dict[str, Any], timestamp: str,
) -> dict[str, float]:
    """Fit + evaluate logistic regression inside a tracked MLflow run."""
    import mlflow  # lazy: keeps mlflow optional for non-training callers

    run_name = f"logistic_regression_{timestamp}"
    with mlflow.start_run(run_name=run_name):
        # -- Parameters --
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("test_size", cfg["data"]["test_size"])
        mlflow.log_param("random_state", cfg["data"]["random_seed"])
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("solver", "lbfgs")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("calibration_method", "none")
        mlflow.log_param("low_threshold", cfg["policy"]["low_threshold"])
        mlflow.log_param("high_threshold", cfg["policy"]["high_threshold"])

        model = train_logistic_regression(
            X_train, y_train, random_state=cfg["data"]["random_seed"]
        )

        # -- Metrics --
        metrics = evaluate_model(model, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k.replace("-", "_"), v)

        _log_portfolio_metrics(
            model, X_test, y_test,
            t_low=cfg["policy"]["low_threshold"],
            t_high=cfg["policy"]["high_threshold"],
        )

        # -- Artifacts --
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model_path = tmp / "logistic_regression.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(model_path), artifact_path="model")

            probs = model.predict_proba(X_test)[:, 1]
            cal_path = tmp / "calibration_curve.png"
            _save_calibration_plot(
                y_test.values, probs, cal_path, "Logistic Regression — Calibration"
            )
            mlflow.log_artifact(str(cal_path), artifact_path="plots")

            fi_path = tmp / "feature_importance.png"
            _save_feature_importance_plot(model, feature_names, fi_path)
            if fi_path.exists():
                mlflow.log_artifact(str(fi_path), artifact_path="plots")

        return metrics


def _run_xgb_experiment(
    X_train, X_test, y_train, y_test, feature_names: list[str],
    cfg: dict[str, Any], timestamp: str,
) -> dict[str, float]:
    """Fit + calibrate + evaluate XGBoost inside a tracked MLflow run.

    The persisted artifact used by the rest of the project (SHAP, scoring,
    fairness notebook) is also overwritten here so ``models/`` stays in sync
    with the best tracked run.
    """
    import mlflow  # lazy: keeps mlflow optional for non-training callers

    run_name = f"xgboost_calibrated_{timestamp}"
    with mlflow.start_run(run_name=run_name):
        xgb_params = cfg["model"]["params"]
        # -- Parameters --
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("test_size", cfg["data"]["test_size"])
        mlflow.log_param("random_state", cfg["data"]["random_seed"])
        mlflow.log_param("n_estimators", xgb_params["n_estimators"])
        mlflow.log_param("max_depth", xgb_params["max_depth"])
        mlflow.log_param("learning_rate", xgb_params["learning_rate"])
        mlflow.log_param("subsample", xgb_params["subsample"])
        mlflow.log_param("colsample_bytree", xgb_params["colsample_bytree"])
        mlflow.log_param("scale_pos_weight", 11)
        mlflow.log_param("calibration_method", "isotonic")
        mlflow.log_param("low_threshold", cfg["policy"]["low_threshold"])
        mlflow.log_param("high_threshold", cfg["policy"]["high_threshold"])

        xgb = train_xgboost(X_train, y_train)
        xgb_cal = calibrate_model(xgb, X_test, y_test)

        # -- Metrics (report calibrated model; raw is inferior on Brier) --
        metrics = evaluate_model(xgb_cal, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k.replace("-", "_"), v)

        _log_portfolio_metrics(
            xgb_cal, X_test, y_test,
            t_low=cfg["policy"]["low_threshold"],
            t_high=cfg["policy"]["high_threshold"],
        )

        # -- Artifacts --
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model_path = tmp / "xgboost_calibrated.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(xgb_cal, f)
            mlflow.log_artifact(str(model_path), artifact_path="model")

            probs_cal = xgb_cal.predict_proba(X_test)[:, 1]
            cal_path = tmp / "calibration_curve.png"
            _save_calibration_plot(
                y_test.values, probs_cal, cal_path,
                "XGBoost (isotonic-calibrated) — Calibration",
            )
            mlflow.log_artifact(str(cal_path), artifact_path="plots")

            fi_path = tmp / "feature_importance.png"
            # Use the raw tree model for importances
            _save_feature_importance_plot(xgb, feature_names, fi_path)
            mlflow.log_artifact(str(fi_path), artifact_path="plots")

        # -- Also refresh the pinned artifact used by downstream code --
        model_dir = repo_root() / cfg["output"]["model_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "xgboost_calibrated.pkl", "wb") as f:
            pickle.dump(xgb_cal, f)

        return metrics


def main() -> dict[str, dict[str, float]]:
    """Run both tracked training experiments back-to-back.

    1. Logistic-regression baseline run.
    2. XGBoost + isotonic-calibration run (refreshes ``models/xgboost_calibrated.pkl``).

    Experiment name and tracking URI come from ``configs/config.yaml``. Each
    run is tagged with a timestamp so the MLflow UI shows them as separate
    comparable entries.
    """
    import mlflow  # lazy: keeps mlflow optional for non-training callers

    cfg = load_config()

    # Resolve tracking_uri to an absolute ``file:`` path (repo-root relative).
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    if not tracking_uri.startswith(("http://", "https://", "file:")):
        tracking_uri = (repo_root() / tracking_uri).as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X_train, X_test, y_train, y_test, feature_names = load_processed_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    lr_metrics = _run_lr_experiment(
        X_train, X_test, y_train, y_test, feature_names, cfg, timestamp
    )
    xgb_metrics = _run_xgb_experiment(
        X_train, X_test, y_train, y_test, feature_names, cfg, timestamp
    )

    metrics = {
        "Logistic Regression": lr_metrics,
        "XGBoost (calibrated)": xgb_metrics,
    }
    comparison = pd.DataFrame(metrics).T
    print("\n=== Model Comparison (Test Set) ===")
    print(comparison.to_string())
    print(f"\nMLflow tracking URI : {tracking_uri}")
    print(f"Experiment name     : {cfg['mlflow']['experiment_name']}")
    return metrics


if __name__ == "__main__":
    main()
