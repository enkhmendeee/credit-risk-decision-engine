"""FastAPI inference service for the credit-risk model.

Exposes four endpoints:
    GET  /health       — liveness + model identity
    POST /score        — single-applicant decision
    POST /score/batch  — up to 1000 applicants per request
    GET  /model/info   — model metadata + thresholds

All feature fields on the request are optional (default ``None``); any
field the client omits is filled in with the median from the training set
before scoring, so partial applicant records are still scorable.

Run locally with::

    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model
from sklearn.preprocessing import LabelEncoder

from .config import load_config, repo_root
from .score import load_model, score_applicant

API_VERSION = "0.1.0"
MAX_BATCH_SIZE = 1000
BAND_TO_DECISION = {"LOW": "APPROVE", "MEDIUM": "REVIEW", "HIGH": "REJECT"}

logger = logging.getLogger("credit_risk.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- Model load at import time -----------------------------------------------
#
# We load the model here (rather than in ``lifespan``) so the Pydantic request
# schema below can be built dynamically from the model's own feature list. The
# cost is a one-time pickle load on import.

_cfg = load_config()
_MODEL_PATH = repo_root() / _cfg["output"]["model_dir"] / "xgboost_calibrated.pkl"

logger.info("loading model from %s", _MODEL_PATH)
MODEL = load_model(_MODEL_PATH)
FEATURE_NAMES: list[str] = list(MODEL.base_model.get_booster().feature_names or [])
if not FEATURE_NAMES and hasattr(MODEL.base_model, "feature_names_in_"):
    FEATURE_NAMES = list(MODEL.base_model.feature_names_in_)


# --- Pydantic schemas ---------------------------------------------------------


# Build ``Applicant`` dynamically: applicant_id is required; every model
# feature becomes an optional field with default ``None``. Unknown fields
# are ignored so callers can pass extra metadata harmlessly.
_applicant_fields: dict[str, Any] = {
    "applicant_id": (int | str, Field(..., description="Client identifier (SK_ID_CURR)")),
}
for _feat in FEATURE_NAMES:
    _applicant_fields[_feat] = (float | int | str | None, Field(default=None))

Applicant = create_model(
    "Applicant",
    __config__=ConfigDict(extra="ignore"),
    **_applicant_fields,
)
Applicant.__doc__ = (
    "Partial applicant record. ``applicant_id`` is required; every feature "
    "field is optional and defaults to ``None``. Missing features are "
    "imputed with the training-set median before scoring."
)


class BatchRequest(BaseModel):
    applicants: list[Applicant] = Field(..., description="Applicants to score")  # type: ignore[valid-type]

    @staticmethod
    def _validate_size(v: list) -> list:
        if not v:
            raise ValueError("applicants must not be empty")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"batch size {len(v)} exceeds limit {MAX_BATCH_SIZE}")
        return v

    def model_post_init(self, __context: Any) -> None:  # pydantic v2 hook
        self._validate_size(self.applicants)


class ScoreResponse(BaseModel):
    applicant_id: int | str
    default_probability: float
    risk_band: str
    decision: str
    adverse_action_reasons: list[str] | None = None
    model_version: str


class BatchResponse(BaseModel):
    scores: list[ScoreResponse]


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    auc_roc: float
    calibration_method: str
    low_threshold: float
    high_threshold: float
    training_date: str
    n_features: int
    model_version: str


# --- Lifespan: compute medians & pin config values ---------------------------


state: dict[str, Any] = {}


def _compute_training_medians() -> dict[str, float]:
    """Compute the median value of each model feature from the engineered CSV.

    The processed CSV has raw string categoricals; we reproduce the training
    pipeline's ``LabelEncoder`` step so medians line up with the numeric space
    the model was fit on.
    """
    csv_path = repo_root() / _cfg["data"]["processed_dir"] / "train_engineered.csv"
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    df = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    return df[FEATURE_NAMES].median().to_dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compute training medians and pin config values for the process lifetime."""
    medians = _compute_training_medians()
    training_date = datetime.fromtimestamp(
        _MODEL_PATH.stat().st_mtime, tz=timezone.utc
    ).isoformat()

    state["model"] = MODEL
    state["feature_names"] = FEATURE_NAMES
    state["medians"] = medians
    state["t_low"] = float(_cfg["policy"]["low_threshold"])
    state["t_high"] = float(_cfg["policy"]["high_threshold"])
    state["training_date"] = training_date
    logger.info(
        "service ready: %d features, thresholds=(%.4f, %.4f), medians cached",
        len(FEATURE_NAMES), state["t_low"], state["t_high"],
    )
    yield
    state.clear()


app = FastAPI(
    title="Credit Risk Decision Engine",
    version=API_VERSION,
    description="Inference service for the calibrated XGBoost credit-risk model.",
    lifespan=lifespan,
)


# --- Helpers ------------------------------------------------------------------


def _fill_missing(raw: dict[str, Any]) -> dict[str, Any]:
    """Replace any ``None`` feature value with the training-set median."""
    medians = state["medians"]
    return {
        f: (raw[f] if raw.get(f) is not None else medians[f])
        for f in state["feature_names"]
    }


def _score_one(applicant: Applicant) -> ScoreResponse:  # type: ignore[valid-type]
    """Run the model on a validated Applicant and wrap the response.

    Pydantic has already validated input by the time this runs; anything
    raised here is server-side and becomes a 500.
    """
    data = applicant.model_dump()
    applicant_id = data.pop("applicant_id")
    features = _fill_missing(data)

    try:
        result = score_applicant(
            features,
            model=state["model"],
            feature_names=state["feature_names"],
            t_low=state["t_low"],
            t_high=state["t_high"],
        )
    except Exception as e:
        logger.exception("model error for applicant_id=%s", applicant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"model error: {e}",
        )

    band = result["risk_band"]
    decision = BAND_TO_DECISION[band]
    logger.info(
        "scored applicant_id=%s band=%s decision=%s prob=%.4f",
        applicant_id, band, decision, result["default_probability"],
    )
    return ScoreResponse(
        applicant_id=applicant_id,
        default_probability=result["default_probability"],
        risk_band=band,
        decision=decision,
        adverse_action_reasons=result["adverse_action_reasons"] if decision == "REJECT" else None,
        model_version=API_VERSION,
    )


# --- Endpoints ----------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe.

    Sample::

        curl http://localhost:8000/health
        {"status":"ok","model":"xgboost_calibrated","version":"0.1.0"}
    """
    return HealthResponse(status="ok", model="xgboost_calibrated", version=API_VERSION)


@app.post("/score", response_model=ScoreResponse)
async def score(applicant: Applicant) -> ScoreResponse:  # type: ignore[valid-type]
    """Score a single applicant and return a decision.

    Every feature field is optional. Omitted or ``None`` values are filled
    with the training-set median before the model runs.

    Sample (partial record)::

        curl -X POST http://localhost:8000/score \\
          -H 'Content-Type: application/json' \\
          -d '{"applicant_id": 100001,
               "EXT_SOURCE_MEAN": 0.42,
               "CREDIT_INCOME_RATIO": 3.1,
               "AGE_YEARS": 35,
               "AMT_CREDIT": 450000}'
    """
    return _score_one(applicant)


@app.post("/score/batch", response_model=BatchResponse)
async def score_batch_endpoint(payload: BatchRequest) -> BatchResponse:
    """Score a list of applicants (max 1000 per request).

    Sample::

        curl -X POST http://localhost:8000/score/batch \\
          -H 'Content-Type: application/json' \\
          -d '{"applicants": [
                 {"applicant_id": 1, "EXT_SOURCE_MEAN": 0.5},
                 {"applicant_id": 2}
               ]}'
    """
    return BatchResponse(scores=[_score_one(a) for a in payload.applicants])


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return model metadata (type, calibration, thresholds, training date).

    Sample::

        curl http://localhost:8000/model/info
    """
    return ModelInfoResponse(
        model_type="xgboost",
        auc_roc=0.77,
        calibration_method="isotonic",
        low_threshold=state["t_low"],
        high_threshold=state["t_high"],
        training_date=state["training_date"],
        n_features=len(state["feature_names"]),
        model_version=API_VERSION,
    )


# --- Global error handler for unhandled exceptions ---------------------------


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Convert any uncaught exception into a clean 500 response.

    Pydantic validation failures are handled by FastAPI's built-in 422 handler,
    so they do not reach this hook.
    """
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "internal server error"},
    )
