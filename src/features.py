"""Feature-engineering pipeline for the Home Credit default-risk dataset.

Lifts the logic from ``notebooks/02_feature_engineering.ipynb`` into a set of
testable functions plus a single ``run_feature_pipeline`` entry point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import load_config, repo_root

# Columns that must never be dropped/imputed as if they were ordinary features.
_PROTECTED_COLS = {"SK_ID_CURR", "TARGET"}

# DAYS_EMPLOYED uses 365243 to encode "retired / never employed" — treat as NaN.
_DAYS_EMPLOYED_ANOMALY = 365243


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw ``application_train.csv`` file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the raw CSV.
    """
    return pd.read_csv(path)


def drop_high_missing_cols(
    df: pd.DataFrame,
    threshold: float = 50.0,
    keep: Iterable[str] = ("EXT_SOURCE_1",),
) -> pd.DataFrame:
    """Drop columns whose missing-value percentage exceeds ``threshold``.

    Columns listed in ``keep`` are always retained (the notebook keeps
    ``EXT_SOURCE_1`` despite being ~56 % null because the bureau score is
    highly predictive and imputing + flagging preserves most of the signal).

    The returned DataFrame also carries an ``EXT_SOURCE_1_missing`` flag and
    has its ``EXT_SOURCE_1`` NaNs filled with the column median.
    """
    keep = set(keep)
    missing_pct = df.isnull().mean() * 100
    to_drop = [c for c in df.columns if missing_pct[c] > threshold and c not in keep]
    df = df.drop(columns=to_drop)

    if "EXT_SOURCE_1" in df.columns:
        df["EXT_SOURCE_1_missing"] = df["EXT_SOURCE_1"].isnull().astype("int8")
        df["EXT_SOURCE_1"] = df["EXT_SOURCE_1"].fillna(df["EXT_SOURCE_1"].median())
    return df


def impute_and_flag(
    df: pd.DataFrame,
    exclude: Iterable[str] = (
        "SK_ID_CURR", "TARGET", "EXT_SOURCE_1", "EXT_SOURCE_1_missing",
    ),
    flag_threshold: float = 20.0,
) -> pd.DataFrame:
    """Impute missing values and add binary missing-indicator columns.

    Numeric columns are filled with the median; object columns with the mode.
    Columns with more than ``flag_threshold``% missing also get a
    ``<col>_missing`` companion column because the missingness pattern itself
    can be predictive.
    """
    exclude = set(exclude)
    missing_pct = df.isnull().mean() * 100

    for col in df.select_dtypes(include="number").columns:
        if col in exclude or missing_pct[col] == 0:
            continue
        if missing_pct[col] > flag_threshold:
            df[col + "_missing"] = df[col].isnull().astype("int8")
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        if col in exclude or missing_pct[col] == 0:
            continue
        if missing_pct[col] > flag_threshold:
            df[col + "_missing"] = df[col].isnull().astype("int8")
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven derived features used by the calibrated model.

    - ``AGE_YEARS`` from ``DAYS_BIRTH`` (positive, in years).
    - ``YEARS_EMPLOYED`` + ``DAYS_EMPLOYED_anomaly`` flag.
    - Financial ratios (credit/income, annuity/income, credit/annuity term,
      employed/life, income per family member).
    - Aggregates of the three external bureau scores (mean, min, weighted).
    - ``DOCUMENT_FLAGS_SUM`` (count of ``FLAG_DOCUMENT_*`` = 1).
    - ``SOCIAL_CIRCLE_DEFAULT_RATE`` (guarded against divide-by-zero).
    """
    # Age / employment
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (df["DAYS_BIRTH"].abs() / 365.25).round(2)
        df = df.drop(columns=["DAYS_BIRTH"])

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_anomaly"] = (
            df["DAYS_EMPLOYED"] == _DAYS_EMPLOYED_ANOMALY
        ).astype("int8")
        days = df["DAYS_EMPLOYED"].replace(_DAYS_EMPLOYED_ANOMALY, np.nan)
        df["YEARS_EMPLOYED"] = (days.abs() / 365.25).round(2).fillna(0)
        df = df.drop(columns=["DAYS_EMPLOYED"])

    # Domain ratios
    df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"]       / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"]      / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"]          = df["AMT_CREDIT"]       / df["AMT_ANNUITY"]
    df["EMPLOYED_LIFE_RATIO"]  = df["YEARS_EMPLOYED"]   / df["AGE_YEARS"]
    df["INCOME_PER_PERSON"]    = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    df["EMPLOYED_LIFE_RATIO"] = df["EMPLOYED_LIFE_RATIO"].clip(0, 1)

    # CNT_FAM_MEMBERS == 0 produces inf — fall back to total income
    mask = df["INCOME_PER_PERSON"].isnull() | np.isinf(df["INCOME_PER_PERSON"])
    df.loc[mask, "INCOME_PER_PERSON"] = df.loc[mask, "AMT_INCOME_TOTAL"]

    for col in [
        "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
        "EMPLOYED_LIFE_RATIO", "INCOME_PER_PERSON",
    ]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # External bureau score aggregates
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"]     = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_MIN"]      = df[ext_cols].min(axis=1)
    df["EXT_SOURCE_WEIGHTED"] = (
        df["EXT_SOURCE_1"] * 1
        + df["EXT_SOURCE_2"] * 2
        + df["EXT_SOURCE_3"] * 3
    ) / 6

    # Document flags (presence of any of the ~20 FLAG_DOCUMENT_* columns)
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCUMENT_FLAGS_SUM"] = df[doc_cols].sum(axis=1)

    # Social circle default rate (guard zero-observation rows)
    if {"OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE"}.issubset(df.columns):
        obs = df["OBS_30_CNT_SOCIAL_CIRCLE"].replace(0, np.nan)
        df["SOCIAL_CIRCLE_DEFAULT_RATE"] = (
            df["DEF_30_CNT_SOCIAL_CIRCLE"] / obs
        ).fillna(0)

    return df


def run_feature_pipeline(
    raw_path: str | Path,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run the end-to-end feature pipeline and persist the result.

    Steps:
        1. load raw CSV
        2. drop columns with >50 % missing (keep ``EXT_SOURCE_1``)
        3. impute + add missing-value flags
        4. engineer domain features

    The output DataFrame is written to ``out_path`` (defaults to
    ``data/processed/train_engineered.csv``) and also returned.
    """
    cfg = load_config()
    if out_path is None:
        out_path = repo_root() / cfg["data"]["processed_dir"] / "train_engineered.csv"
    out_path = Path(out_path)

    df = load_raw_data(raw_path)
    df = drop_high_missing_cols(df)
    df = impute_and_flag(df)
    df = engineer_features(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    cfg = load_config()
    raw = repo_root() / cfg["data"]["raw_dir"] / "application_train.csv"
    out = run_feature_pipeline(raw)
    print(f"Engineered dataset saved with shape {out.shape}")
