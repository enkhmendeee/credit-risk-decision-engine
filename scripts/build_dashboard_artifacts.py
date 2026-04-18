"""Produce the four dashboard artifacts consumed by the Streamlit app.

Outputs:
    reports/calibration_curve.png
    reports/feature_importance.csv
    data/test_scores.csv
    data/fairness_metrics.csv

Uses the same 80/20 stratified split, label encoding, and calibrated XGBoost
model as notebooks 03 / 05 / 06, so scores here line up with the notebooks.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import load_config, repo_root
from src.policy import assign_risk_band
from src.train import IsotonicCalibratedModel  # noqa: F401  (pickle needs it)


SHAP_SAMPLE_N = 5000
RANDOM_STATE = 42


def load_split_and_model():
    cfg = load_config()
    root = repo_root()
    df_path = root / cfg["data"]["processed_dir"] / "train_engineered.csv"
    model_path = root / cfg["output"]["model_dir"] / "xgboost_calibrated.pkl"

    df = pd.read_csv(df_path)

    gender_raw = df["CODE_GENDER"].copy()
    edu_raw = df["NAME_EDUCATION_TYPE"].copy()
    sk_id = df["SK_ID_CURR"].copy()

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        stratify=y,
        random_state=cfg["data"]["random_seed"],
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    t_low = float(cfg["policy"]["low_threshold"])
    t_high = float(cfg["policy"]["high_threshold"])

    return {
        "cfg": cfg,
        "root": root,
        "X_test": X_test,
        "y_test": y_test,
        "gender_test": gender_raw.loc[X_test.index],
        "edu_test": edu_raw.loc[X_test.index],
        "sk_id_test": sk_id.loc[X_test.index],
        "feature_names": X.columns.tolist(),
        "model": model,
        "t_low": t_low,
        "t_high": t_high,
    }


def save_calibration_curve(ctx, probs, out_path: Path) -> None:
    y_test = ctx["y_test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    frac_pos, mean_pred = calibration_curve(
        y_test, probs, n_bins=15, strategy="quantile"
    )
    axes[0].plot(mean_pred, frac_pos, marker="o", lw=2, label="XGBoost (calibrated)")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Fraction of positives")
    axes[0].set_title("Reliability Diagram")
    axes[0].legend()

    axes[1].hist(probs, bins=50, density=True, alpha=0.7)
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Score Distribution")

    fig.suptitle("XGBoost (isotonic-calibrated) — Calibration", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(ctx, out_path: Path) -> None:
    model = ctx["model"]
    X_test = ctx["X_test"]
    feature_names = ctx["feature_names"]

    rng = np.random.default_rng(RANDOM_STATE)
    n = min(SHAP_SAMPLE_N, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_sample = X_test.iloc[idx]

    explainer = shap.TreeExplainer(model.base_model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs = np.abs(shap_values).mean(axis=0)
    top = (
        pd.Series(mean_abs, index=feature_names)
        .sort_values(ascending=False)
        .head(20)
        .rename_axis("feature")
        .reset_index(name="mean_abs_shap")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_path, index=False)


def save_test_scores(ctx, probs, out_path: Path) -> None:
    t_low, t_high = ctx["t_low"], ctx["t_high"]
    bands = np.asarray(assign_risk_band(probs, t_low=t_low, t_high=t_high))
    decision = np.where(bands == "HIGH", "DECLINE", "APPROVE")

    out = pd.DataFrame({
        "SK_ID_CURR": ctx["sk_id_test"].values,
        "y_true": ctx["y_test"].values,
        "y_prob": probs,
        "risk_band": bands,
        "decision": decision,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def _fairness_rows(group_series, approved, y_true, attribute_label, min_n=100):
    rows = []
    for g, idx in group_series.groupby(group_series).groups.items():
        yt = y_true.loc[idx]
        app = approved.loc[idx]
        n = len(idx)
        if n < min_n:
            continue
        n_good = int((yt == 0).sum())
        n_bad = int((yt == 1).sum())
        approval_rate = float(app.mean())
        fpr = float(1 - app[yt == 0].mean()) if n_good else 0.0
        fnr = float(app[yt == 1].mean()) if n_bad else 0.0
        rows.append({
            "attribute": attribute_label,
            "group": g,
            "n_good": n_good,
            "n_bad": n_bad,
            "approval_rate": approval_rate,
            "FPR": fpr,
            "FNR": fnr,
        })
    return rows


def save_fairness_metrics(ctx, probs, out_path: Path) -> None:
    t_low = ctx["t_low"]
    bands = np.asarray(assign_risk_band(probs, t_low=t_low, t_high=ctx["t_high"]))
    approved = pd.Series(bands != "HIGH", index=ctx["X_test"].index)
    y_true = pd.Series(ctx["y_test"].values, index=ctx["X_test"].index)

    rows = []
    rows.extend(_fairness_rows(ctx["gender_test"], approved, y_true, "gender"))
    rows.extend(_fairness_rows(ctx["edu_test"], approved, y_true, "education"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    ctx = load_split_and_model()
    probs = ctx["model"].predict_proba(ctx["X_test"])[:, 1]

    root = ctx["root"]
    save_calibration_curve(ctx, probs, root / "reports" / "calibration_curve.png")
    print("wrote reports/calibration_curve.png")

    save_feature_importance(ctx, root / "reports" / "feature_importance.csv")
    print("wrote reports/feature_importance.csv")

    save_test_scores(ctx, probs, root / "data" / "test_scores.csv")
    print("wrote data/test_scores.csv")

    save_fairness_metrics(ctx, probs, root / "data" / "fairness_metrics.csv")
    print("wrote data/fairness_metrics.csv")


if __name__ == "__main__":
    main()
