"""Risk-policy layer: bands, portfolio simulation, threshold sweep.

Lifts the logic from ``notebooks/04_risk_policy.ipynb`` into importable
functions. Thresholds default to the calibrated-model values used in the
notebook (LOW < 0.30, HIGH > 0.60) but every function takes explicit
arguments so the policy can be re-tuned without rewriting code.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import load_config

# Default band cutoffs (calibrated-probability space). Read from config at
# import time so a single source of truth lives in ``configs/config.yaml``.
try:
    _cfg = load_config()
    DEFAULT_T_LOW = float(_cfg["policy"]["low_threshold"])
    DEFAULT_T_HIGH = float(_cfg["policy"]["high_threshold"])
except (FileNotFoundError, KeyError):
    DEFAULT_T_LOW = 0.30
    DEFAULT_T_HIGH = 0.60


def assign_risk_band(
    probability: float | np.ndarray | pd.Series,
    t_low: float = DEFAULT_T_LOW,
    t_high: float = DEFAULT_T_HIGH,
) -> str | np.ndarray:
    """Map a calibrated default probability to a risk band label.

    Returns ``'LOW'`` if ``p < t_low``, ``'HIGH'`` if ``p > t_high``, and
    ``'MEDIUM'`` otherwise. Accepts either a scalar or an array / Series and
    returns the same shape.
    """
    if np.isscalar(probability):
        p = float(probability)
        if p < t_low:
            return "LOW"
        if p > t_high:
            return "HIGH"
        return "MEDIUM"

    arr = np.asarray(probability)
    out = np.full(arr.shape, "MEDIUM", dtype=object)
    out[arr < t_low] = "LOW"
    out[arr > t_high] = "HIGH"
    return out


def simulate_portfolio(
    probabilities: np.ndarray | pd.Series,
    actuals: np.ndarray | pd.Series,
    loan_amounts: np.ndarray | pd.Series,
    t_low: float = DEFAULT_T_LOW,
    t_high: float = DEFAULT_T_HIGH,
) -> pd.DataFrame:
    """Simulate portfolio outcomes under two approval policies.

    Returns a DataFrame indexed by scenario (``Conservative``, ``Moderate``)
    with columns for approved count, approval rate, portfolio default rate,
    expected defaults, and expected loss (using each approved loan's actual
    ``AMT_CREDIT`` rather than a fleet average).

    - **Conservative** approves only LOW-band applicants.
    - **Moderate** approves LOW + MEDIUM applicants (everything below
      ``t_high``).
    """
    df = pd.DataFrame({
        "prob": np.asarray(probabilities),
        "actual": np.asarray(actuals),
        "AMT_CREDIT": np.asarray(loan_amounts),
    })
    df["band"] = assign_risk_band(df["prob"], t_low=t_low, t_high=t_high)

    scenarios = {
        "Conservative (LOW only)":  df["band"] == "LOW",
        "Moderate (LOW + MEDIUM)":  df["band"].isin(["LOW", "MEDIUM"]),
    }

    total = len(df)
    rows = []
    for name, mask in scenarios.items():
        approved = df[mask]
        n = len(approved)
        portfolio_dr = float(approved["actual"].mean()) if n else 0.0
        expected_defaults = n * portfolio_dr
        expected_loss = float((approved["actual"] * approved["AMT_CREDIT"]).sum())
        rows.append({
            "scenario": name,
            "n_approved": n,
            "approval_rate": n / total if total else 0.0,
            "portfolio_default_rate": portfolio_dr,
            "expected_defaults": expected_defaults,
            "expected_loss": expected_loss,
        })
    return pd.DataFrame(rows).set_index("scenario")


def threshold_analysis(
    probabilities: np.ndarray | pd.Series,
    actuals: np.ndarray | pd.Series,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Sweep decision thresholds and return per-threshold business metrics.

    For each threshold ``t``, applicants with ``prob < t`` are approved.
    The returned DataFrame includes approval rate, portfolio default rate
    (default rate among approved), recall on defaulters (1 - recall of the
    decline class), and F1 of the decline decision.

    If ``thresholds`` is None, uses a dense grid ``0.05 .. 0.95`` at 0.05
    spacing — the same grid the notebook uses.
    """
    probs = np.asarray(probabilities)
    actuals = np.asarray(actuals)
    if thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.05)

    rows = []
    total = len(probs)
    n_default = int(actuals.sum())
    for t in thresholds:
        approve_mask = probs < t
        n_approved = int(approve_mask.sum())
        approval_rate = n_approved / total if total else 0.0
        portfolio_dr = float(actuals[approve_mask].mean()) if n_approved else 0.0
        # "decline" class: label = 1 (defaulter), predicted = 1 when prob >= t
        decline_pred = (~approve_mask).astype(int)
        recall = (
            float((decline_pred[actuals == 1]).mean()) if n_default else 0.0
        )
        f1 = f1_score(actuals, decline_pred, zero_division=0)
        rows.append({
            "threshold": round(float(t), 4),
            "approval_rate": approval_rate,
            "portfolio_default_rate": portfolio_dr,
            "recall_defaulters": recall,
            "f1": float(f1),
        })
    return pd.DataFrame(rows)
