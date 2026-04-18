"""Scoring layer: single-applicant and batch inference.

Lifts the scoring + adverse-action logic from
``notebooks/05_explainability.ipynb`` into reusable functions. Callers pass
a pre-processed applicant record (same feature schema as training); this
module is not responsible for feature engineering (see ``features.py``).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, repo_root
from .policy import DEFAULT_T_HIGH, DEFAULT_T_LOW, assign_risk_band
# Re-export so ``pickle.load`` can resolve the wrapper class from this module.
from .train import IsotonicCalibratedModel  # noqa: F401

# --- Adverse-action configuration (mirrors notebooks/05_explainability) ---

# Features forbidden in adverse-action reasons: protected or close proxies.
EXCLUDED_FROM_REASONS: set[str] = {
    "CODE_GENDER", "CODE_GENDER_missing",
    "NAME_FAMILY_STATUS", "NAME_FAMILY_STATUS_missing",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "NAME_HOUSING_TYPE", "NAME_HOUSING_TYPE_missing",
    "REGION_POPULATION_RELATIVE",
    "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
}

REASON_TEMPLATES: dict[str, str] = {
    "EXT_SOURCE_MEAN":            "Low external credit score",
    "EXT_SOURCE_MIN":             "Low minimum external credit score",
    "EXT_SOURCE_WEIGHTED":        "Low weighted external credit score",
    "EXT_SOURCE_1":               "Low credit bureau score (source 1)",
    "EXT_SOURCE_2":               "Low credit bureau score (source 2)",
    "EXT_SOURCE_3":               "Low credit bureau score (source 3)",
    "CREDIT_INCOME_RATIO":        "High debt-to-income ratio",
    "ANNUITY_INCOME_RATIO":       "High repayment burden relative to income",
    "CREDIT_TERM":                "Long implied loan term",
    "EMPLOYED_LIFE_RATIO":        "Short employment history relative to age",
    "INCOME_PER_PERSON":          "Low household income per person",
    "YEARS_EMPLOYED":             "Short employment history",
    "AGE_YEARS":                  "Applicant age",
    "AMT_CREDIT":                 "Requested loan amount",
    "AMT_INCOME_TOTAL":           "Income level",
    "AMT_ANNUITY":                "Monthly repayment amount",
    "AMT_GOODS_PRICE":            "Goods price",
    "DAYS_EMPLOYED_anomaly":      "Irregular employment record",
    "DOCUMENT_FLAGS_SUM":         "Insufficient documentation provided",
    "SOCIAL_CIRCLE_DEFAULT_RATE": "High default rate in social circle",
    "REGION_RATING_CLIENT":       "High-risk residential region",
    "EXT_SOURCE_1_missing":       "Missing credit bureau record (source 1)",
}

_CURRENCY_FEATURES = {
    "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "INCOME_PER_PERSON",
}
_AGE_FEATURES = {"AGE_YEARS"}
_RATIO_FEATURES = {
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
    "EMPLOYED_LIFE_RATIO", "SOCIAL_CIRCLE_DEFAULT_RATE",
    "EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "EXT_SOURCE_WEIGHTED",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
}


def _format_value(feat: str, value: float) -> str:
    """Pretty-print a feature value based on its semantic type."""
    if feat in _CURRENCY_FEATURES:
        return f"${value:,.0f}"
    if feat in _AGE_FEATURES:
        return f"{int(round(value))} years"
    if feat in _RATIO_FEATURES:
        return f"{value:.2f}"
    return f"{value:.4g}"


def _format_reason(feat: str, value: float) -> str:
    label = REASON_TEMPLATES.get(feat, feat.replace("_", " ").title())
    return f"{label} ({feat} = {_format_value(feat, value)})"


def load_model(path: str | Path | None = None):
    """Load the calibrated model from disk.

    Defaults to ``models/xgboost_calibrated.pkl`` resolved from the repo root.
    The returned object is an :class:`IsotonicCalibratedModel` that exposes
    both calibrated probabilities (via ``predict_proba``) and the underlying
    tree model (``base_model``) for SHAP.
    """
    if path is None:
        cfg = load_config()
        path = repo_root() / cfg["output"]["model_dir"] / "xgboost_calibrated.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _top_shap_reasons(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: pd.Series,
    top_n: int = 3,
) -> list[str]:
    """Pick the top ``top_n`` features with the largest positive SHAP values,
    excluding protected/demographic columns, and format each as a reason line.
    """
    contrib = pd.Series(shap_values, index=feature_names)
    eligible = contrib[(contrib > 0) & (~contrib.index.isin(EXCLUDED_FROM_REASONS))]
    drivers = eligible.nlargest(top_n)
    return [
        _format_reason(feat, float(feature_values[feat])) for feat in drivers.index
    ]


def score_applicant(
    applicant_dict: dict[str, Any],
    model=None,
    feature_names: list[str] | None = None,
    explainer=None,
    t_low: float = DEFAULT_T_LOW,
    t_high: float = DEFAULT_T_HIGH,
) -> dict[str, Any]:
    """Score a single already-engineered applicant record.

    Parameters
    ----------
    applicant_dict
        Mapping ``feature_name -> value``. Must include every feature the
        model was trained on (label-encoded for categoricals).
    model
        Calibrated model. If ``None``, loads the default artifact.
    feature_names
        Column order expected by the model. Defaults to
        ``applicant_dict.keys()`` but passing an explicit list is strongly
        recommended to guarantee column alignment with training.
    explainer
        A fitted ``shap.TreeExplainer`` on ``model.base_model``. If the
        applicant lands in the HIGH band and ``explainer`` is None, one is
        created on the fly (requires ``shap``).

    Returns
    -------
    dict with keys:
        - ``default_probability`` (float, calibrated)
        - ``risk_band`` (``'LOW'`` / ``'MEDIUM'`` / ``'HIGH'``)
        - ``adverse_action_reasons`` (list of str, empty unless HIGH)
    """
    if model is None:
        model = load_model()
    if feature_names is None:
        feature_names = list(applicant_dict.keys())

    x = pd.DataFrame([[applicant_dict[f] for f in feature_names]], columns=feature_names)
    prob = float(model.predict_proba(x)[:, 1][0])
    band = assign_risk_band(prob, t_low=t_low, t_high=t_high)

    reasons: list[str] = []
    if band == "HIGH":
        if explainer is None:
            import shap  # local import keeps the dependency optional
            explainer = shap.TreeExplainer(model.base_model)
        shap_vals = np.asarray(explainer.shap_values(x))[0]
        reasons = _top_shap_reasons(shap_vals, feature_names, x.iloc[0], top_n=3)

    return {
        "default_probability": prob,
        "risk_band": band,
        "adverse_action_reasons": reasons,
    }


def score_batch(
    df: pd.DataFrame,
    model=None,
    t_low: float = DEFAULT_T_LOW,
    t_high: float = DEFAULT_T_HIGH,
) -> pd.DataFrame:
    """Score every row of an already-engineered DataFrame.

    Returns a DataFrame aligned with ``df.index`` containing
    ``default_probability`` and ``risk_band``. Adverse-action reasons are
    **not** computed in batch mode (cost-prohibitive: one SHAP call per row).
    Use :func:`score_applicant` for full reason codes on rejected applicants.
    """
    if model is None:
        model = load_model()

    probs = model.predict_proba(df)[:, 1]
    bands = assign_risk_band(probs, t_low=t_low, t_high=t_high)
    return pd.DataFrame(
        {"default_probability": probs, "risk_band": bands}, index=df.index
    )
