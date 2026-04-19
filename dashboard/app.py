"""Streamlit analyst dashboard for the Credit Risk Decision Engine.

Four pages (sidebar-navigated):
    1. Model Overview        — KPIs, calibration, global feature importance
    2. Score Applicant       — form-driven single-applicant decision + SHAP
    3. Portfolio Analytics   — band distribution, threshold sweep, simulation
    4. Fairness Report       — group-wise approval, FPR, FNR with caveats

Launch:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.policy import (  # noqa: E402
    assign_risk_band,
    simulate_portfolio,
    threshold_analysis,
)
from src.score import load_model, score_applicant  # noqa: E402

# --- Paths & constants --------------------------------------------------------

CFG = load_config()

STREAMLIT_ARTIFACTS = REPO_ROOT / "streamlit_artifacts"


def _resolve(filename: str, fallback: Path) -> Path:
    """Prefer the bundled streamlit_artifacts copy (for cloud deploys); fall back to the original."""
    bundled = STREAMLIT_ARTIFACTS / filename
    return bundled if bundled.exists() else fallback


PATHS = {
    "calibration_curve":  _resolve("calibration_curve.png",  REPO_ROOT / "reports" / "calibration_curve.png"),
    "feature_importance": _resolve("feature_importance.csv", REPO_ROOT / "reports" / "feature_importance.csv"),
    "test_scores":        _resolve("test_scores.csv",        REPO_ROOT / "data" / "test_scores.csv"),
    "fairness_metrics":   _resolve("fairness_metrics.csv",   REPO_ROOT / "data" / "fairness_metrics.csv"),
    "processed_data":     REPO_ROOT / CFG["data"]["processed_dir"] / "train_engineered.csv",
    "medians":            _resolve("feature_medians.json",   REPO_ROOT / CFG["output"]["model_dir"] / "feature_medians.json"),
    "model":              _resolve("xgboost_calibrated.pkl", REPO_ROOT / CFG["output"]["model_dir"] / "xgboost_calibrated.pkl"),
}

T_LOW  = float(CFG["policy"]["low_threshold"])
T_HIGH = float(CFG["policy"]["high_threshold"])

BAND_DECISION = {"LOW": "APPROVE", "MEDIUM": "REVIEW", "HIGH": "REJECT"}
DECISION_COLOR = {
    "APPROVE": "#2ecc71",
    "REVIEW":  "#f39c12",
    "REJECT":  "#e74c3c",
}
BAND_COLOR = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}

# LabelEncoder maps string → integer by sorted unique value. The encodings
# below match `train.py` / `05_explainability.ipynb` for the two categorical
# inputs exposed on the scoring form.
GENDER_ENCODING    = {"F": 0, "M": 1}
EDUCATION_OPTIONS  = [
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
EDUCATION_ENCODING = {
    "Academic degree":               0,
    "Higher education":              1,
    "Incomplete higher":             2,
    "Lower secondary":               3,
    "Secondary / secondary special": 4,
}


# --- Cached loaders -----------------------------------------------------------


@st.cache_resource(show_spinner="Loading calibrated model…")
def load_model_bundle():
    """Load the calibrated model, feature list, SHAP explainer, and medians."""
    model = load_model(PATHS["model"])
    booster_feats = list(model.base_model.get_booster().feature_names or [])
    if not booster_feats and hasattr(model.base_model, "feature_names_in_"):
        booster_feats = list(model.base_model.feature_names_in_)
    with open(PATHS["medians"], "r") as f:
        medians = {k: float(v) for k, v in json.load(f).items()}
    explainer = shap.TreeExplainer(model.base_model)
    return {
        "model":         model,
        "feature_names": booster_feats,
        "medians":       medians,
        "explainer":     explainer,
    }


@st.cache_data(show_spinner=False)
def load_csv(path_key: str) -> pd.DataFrame | None:
    path = PATHS[path_key]
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_image_bytes(path_key: str) -> bytes | None:
    path = PATHS[path_key]
    if not path.exists():
        return None
    return path.read_bytes()


@st.cache_data(show_spinner="Joining loan amounts for portfolio simulation…")
def load_test_scores_with_credit() -> pd.DataFrame | None:
    """Join test_scores.csv with AMT_CREDIT from the processed training data."""
    scores = load_csv("test_scores")
    if scores is None:
        return None
    if not PATHS["processed_data"].exists():
        return scores
    proc = pd.read_csv(PATHS["processed_data"], usecols=["SK_ID_CURR", "AMT_CREDIT"])
    return scores.merge(proc, on="SK_ID_CURR", how="left")


# --- Small UI helpers ---------------------------------------------------------


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def fmt_currency(x: float) -> str:
    return f"${x:,.0f}"


def warn_missing(path_key: str) -> None:
    st.warning(f"Expected artifact not found: `{PATHS[path_key].relative_to(REPO_ROOT)}`.")


def decision_badge(decision: str, probability: float) -> None:
    color = DECISION_COLOR.get(decision, "#7f8c8d")
    st.markdown(
        f"""
        <div style="
            background:{color};
            color:white;
            padding:18px 24px;
            border-radius:10px;
            text-align:center;
            margin:12px 0 6px 0;
        ">
            <div style="font-size:36px; font-weight:700; letter-spacing:1px;">
                {decision}
            </div>
            <div style="font-size:15px; opacity:.9;">
                predicted default probability
                <b>{probability:.2%}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Page 1: Model Overview ---------------------------------------------------


def page_overview() -> None:
    st.title("Model Overview")
    st.caption(
        "Calibrated XGBoost classifier for Home Credit default risk. "
        "Trained on ~307k applicants, isotonic-calibrated, evaluated on a "
        "stratified 20% hold-out."
    )

    # --- KPI cards ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AUC-ROC",            "0.7712", help="Area under the ROC curve — ranking quality.")
    k2.metric("KS statistic",       "0.4045", help="Max separation between default and non-default CDFs.")
    k3.metric("Brier score",        "0.0669", help="Mean squared calibration error (lower is better).")
    k4.metric("Default base rate",  "8.1%",    help="Fraction of applicants in the raw data that defaulted.")

    st.markdown(
        "**How to read these numbers.** The AUC of 0.77 means that, for a "
        "random pair of one defaulter and one non-defaulter, the model "
        "assigns a higher risk score to the defaulter 77% of the time. "
        "The Brier score after isotonic calibration is close to the theoretical "
        "floor for an 8% base rate — predicted probabilities are reliable "
        "enough to use in expected-loss and pricing calculations. KS of 0.40 "
        "confirms strong separation for portfolio-level decisions."
    )

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Calibration")
        img_bytes = load_image_bytes("calibration_curve")
        if img_bytes is None:
            warn_missing("calibration_curve")
        else:
            st.image(img_bytes, caption="Reliability diagram + score distribution (test set)")

    with col_right:
        st.subheader("Class distribution")
        donut = go.Figure(
            data=[go.Pie(
                labels=["Non-default (91%)", "Default (9%)"],
                values=[91, 9],
                hole=0.55,
                marker=dict(colors=[DECISION_COLOR["APPROVE"], DECISION_COLOR["REJECT"]]),
                textinfo="label+percent",
            )]
        )
        donut.update_layout(
            showlegend=False,
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(donut, width="stretch")
        st.caption(
            "Strongly imbalanced: XGBoost uses `scale_pos_weight=11` to "
            "offset the minority class during training."
        )

    st.divider()

    st.subheader("Top 15 Features — Global SHAP Importance")
    fi = load_csv("feature_importance")
    if fi is None:
        warn_missing("feature_importance")
    else:
        top15 = fi.sort_values("mean_abs_shap", ascending=False).head(15)
        fig = px.bar(
            top15.iloc[::-1],  # reverse so largest sits on top
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            height=520,
            xaxis_title="Mean |SHAP value|",
            yaxis_title="",
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "External credit-bureau scores (`EXT_SOURCE_*`) dominate. "
            "Financial-ratio features built in `src/features.py` — credit "
            "term, credit-to-income — fill most of the remaining signal."
        )


# --- Page 2: Score Applicant --------------------------------------------------


def build_applicant_payload(bundle: dict, form: dict) -> dict:
    """Compose a full feature dict from form inputs + training medians.

    Derived features (`EXT_SOURCE_MEAN/MIN/WEIGHTED`, ratios) are recomputed
    from the user-supplied raw inputs so the model sees internally consistent
    values, not a mix of user data and median-imputed derivatives.
    """
    payload = {f: bundle["medians"][f] for f in bundle["feature_names"]}

    ext1, ext2, ext3 = form["EXT_SOURCE_1"], form["EXT_SOURCE_2"], form["EXT_SOURCE_3"]
    credit, annuity, income = form["AMT_CREDIT"], form["AMT_ANNUITY"], form["AMT_INCOME_TOTAL"]

    direct = {
        "EXT_SOURCE_1":     ext1,
        "EXT_SOURCE_2":     ext2,
        "EXT_SOURCE_3":     ext3,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT":       credit,
        "AMT_ANNUITY":      annuity,
        "AGE_YEARS":        form["AGE_YEARS"],
        "CODE_GENDER":      GENDER_ENCODING[form["CODE_GENDER"]],
        "NAME_EDUCATION_TYPE": EDUCATION_ENCODING[form["NAME_EDUCATION_TYPE"]],
    }
    derived = {
        "EXT_SOURCE_MEAN":     float(np.mean([ext1, ext2, ext3])),
        "EXT_SOURCE_MIN":      float(np.min([ext1, ext2, ext3])),
        "EXT_SOURCE_WEIGHTED": (ext1 * 1 + ext2 * 2 + ext3 * 3) / 6,
        "EXT_SOURCE_1_missing":    0,
        "CREDIT_INCOME_RATIO":  credit  / income  if income  else np.nan,
        "ANNUITY_INCOME_RATIO": annuity / income  if income  else np.nan,
        "CREDIT_TERM":          credit  / annuity if annuity else np.nan,
    }
    for k, v in {**direct, **derived}.items():
        if k in payload and v is not None and not (isinstance(v, float) and np.isnan(v)):
            payload[k] = v
    return payload


def render_shap_waterfall(bundle: dict, payload: dict) -> None:
    """Try shap.plots.waterfall; fall back to a simple matplotlib bar chart."""
    feature_names = bundle["feature_names"]
    x = pd.DataFrame([[payload[f] for f in feature_names]], columns=feature_names)

    try:
        shap_vals = bundle["explainer"].shap_values(x)[0]
        expl = shap.Explanation(
            values=shap_vals,
            base_values=float(bundle["explainer"].expected_value),
            data=x.iloc[0].values,
            feature_names=feature_names,
        )
        fig = plt.figure(figsize=(9, 5))
        shap.plots.waterfall(expl, max_display=12, show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception as err:
        st.info(f"SHAP waterfall rendering failed ({err}). Showing a static bar fallback.")
        shap_vals = bundle["explainer"].shap_values(x)[0]
        top = (
            pd.Series(shap_vals, index=feature_names)
            .reindex(pd.Series(shap_vals, index=feature_names).abs()
                     .sort_values(ascending=False).index)
            .head(12)
            .iloc[::-1]
        )
        colors = [DECISION_COLOR["REJECT"] if v > 0 else DECISION_COLOR["APPROVE"]
                  for v in top.values]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(top.index, top.values, color=colors, edgecolor="white")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP contribution (log-odds)")
        ax.set_title("Top 12 feature contributions")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)


def page_score() -> None:
    st.title("Score Applicant")
    st.caption(
        "Enter an applicant profile and get the calibrated decision back. "
        "Unspecified features use the training-set median — the same imputation "
        "the production `/score` endpoint performs."
    )

    bundle = load_model_bundle()

    with st.form("applicant_form", clear_on_submit=False):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**External credit scores**")
            ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.50, step=0.01)
            ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.50, step=0.01)
            ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.50, step=0.01)
            age  = st.slider("Age (years)",    18, 70, 35)

        with c2:
            st.markdown("**Loan & applicant data**")
            income  = st.number_input("AMT_INCOME_TOTAL ($)", min_value=20_000,
                                      max_value=2_000_000, value=150_000, step=5_000)
            credit  = st.number_input("AMT_CREDIT ($)",       min_value=20_000,
                                      max_value=4_000_000, value=500_000, step=10_000)
            annuity = st.number_input("AMT_ANNUITY ($/yr)",   min_value=2_000,
                                      max_value=300_000,   value=24_000, step=1_000)
            gender  = st.selectbox("CODE_GENDER", ["F", "M"], index=0)
            education = st.selectbox("NAME_EDUCATION_TYPE", EDUCATION_OPTIONS, index=0)

        submitted = st.form_submit_button("Score applicant", width="stretch")

    if not submitted:
        st.info("Fill the form above and submit to see the decision.")
        return

    form = {
        "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
        "AGE_YEARS": age,
        "DAYS_BIRTH": -int(round(age * 365.25)),  # informational; dropped before scoring
        "CODE_GENDER": gender, "NAME_EDUCATION_TYPE": education,
    }

    payload = build_applicant_payload(bundle, form)

    try:
        result = score_applicant(
            payload,
            model=bundle["model"],
            feature_names=bundle["feature_names"],
            explainer=bundle["explainer"],
            t_low=T_LOW,
            t_high=T_HIGH,
        )
    except Exception as err:
        st.error(f"Model scoring failed: {err}")
        return

    band     = result["risk_band"]
    prob     = result["default_probability"]
    decision = BAND_DECISION[band]

    decision_badge(decision, prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk band", band)
    c2.metric("Probability", f"{prob:.2%}")
    c3.metric("Threshold (LOW / HIGH)", f"{T_LOW:.3f} / {T_HIGH:.3f}")

    if decision in ("REJECT", "REVIEW"):
        reasons = result.get("adverse_action_reasons") or []
        if reasons:
            st.subheader("Top contributing risk factors")
            for i, r in enumerate(reasons, 1):
                st.markdown(f"{i}. {r}")
        elif decision == "REVIEW":
            st.caption("Medium-risk cases go to manual review — no adverse-action notice required.")

    st.divider()
    st.subheader("SHAP explanation")
    st.caption(
        "Features pushing the score *up* (toward default) are red; features "
        "pushing the score *down* are green. Values are on the model's log-odds "
        "scale; isotonic calibration preserves their ranking."
    )
    render_shap_waterfall(bundle, payload)

    with st.expander("Raw model input payload"):
        st.json(form)


# --- Page 3: Portfolio Analytics ----------------------------------------------


def page_portfolio() -> None:
    st.title("Portfolio Analytics")
    st.caption(
        "How the model's risk bands translate into portfolio-level outcomes "
        "on the held-out test set (~61.5k applicants)."
    )

    df = load_test_scores_with_credit()
    if df is None:
        warn_missing("test_scores")
        return

    # ---- Decision band distribution ----
    st.subheader("Risk-band distribution")
    band_counts = (
        df["risk_band"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH"])
        .fillna(0)
        .astype(int)
    )
    band_pcts = band_counts / band_counts.sum()
    fig = go.Figure(
        data=[go.Bar(
            x=band_counts.index,
            y=band_counts.values,
            marker_color=[BAND_COLOR[b] for b in band_counts.index],
            text=[f"{n:,} ({p:.1%})" for n, p in zip(band_counts.values, band_pcts.values)],
            textposition="outside",
        )]
    )
    fig.update_layout(
        yaxis_title="Applicants",
        xaxis_title="Risk band",
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"At the configured thresholds (LOW < {T_LOW:.3f}, HIGH > {T_HIGH:.3f}) "
        f"{band_pcts.get('LOW', 0):.1%} of applicants fall in the auto-approve band "
        f"and {band_pcts.get('HIGH', 0):.1%} are auto-rejected; the rest need human review."
    )

    st.divider()

    # ---- Threshold tradeoff curve ----
    st.subheader("Threshold tradeoff — approval rate vs portfolio default rate")
    sweep = threshold_analysis(df["y_prob"].values, df["y_true"].values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep["threshold"], y=sweep["approval_rate"],
        name="Approval rate", mode="lines+markers",
        line=dict(color=DECISION_COLOR["APPROVE"], width=3),
    ))
    fig.add_trace(go.Scatter(
        x=sweep["threshold"], y=sweep["portfolio_default_rate"],
        name="Portfolio default rate", mode="lines+markers",
        line=dict(color=DECISION_COLOR["REJECT"], width=3),
    ))
    fig.add_vline(x=T_LOW,  line_dash="dash", line_color="#666",
                  annotation_text=f"LOW={T_LOW:.3f}",  annotation_position="top left")
    fig.add_vline(x=T_HIGH, line_dash="dash", line_color="#666",
                  annotation_text=f"HIGH={T_HIGH:.3f}", annotation_position="top right")
    fig.update_layout(
        xaxis_title="Decision threshold (probability cutoff for approval)",
        yaxis_title="Rate",
        yaxis_tickformat=".0%",
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Moving the approval cutoff right grows the book but lets more defaulters "
        "in. The configured LOW cutoff keeps realised default inside the approved "
        "portfolio near the target range; the HIGH cutoff defines the auto-decline line."
    )

    st.divider()

    # ---- Expected loss comparison + simulation table ----
    st.subheader("Conservative vs Moderate approval policy")

    if "AMT_CREDIT" not in df.columns or df["AMT_CREDIT"].isna().all():
        st.warning(
            "`data/processed/train_engineered.csv` not found — expected-loss "
            "simulation needs per-applicant AMT_CREDIT. Run the feature pipeline "
            "to produce it."
        )
        return

    sim = simulate_portfolio(
        df["y_prob"].values,
        df["y_true"].values,
        df["AMT_CREDIT"].fillna(df["AMT_CREDIT"].median()).values,
        t_low=T_LOW,
        t_high=T_HIGH,
    )

    bar_fig = go.Figure(
        data=[go.Bar(
            x=sim.index,
            y=sim["expected_loss"].values,
            marker_color=[DECISION_COLOR["APPROVE"], DECISION_COLOR["REVIEW"]],
            text=[fmt_currency(v) for v in sim["expected_loss"].values],
            textposition="outside",
        )]
    )
    bar_fig.update_layout(
        yaxis_title="Expected loss ($, sum over defaults in approved book)",
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(bar_fig, width="stretch")

    st.markdown("**Portfolio simulation table**")
    sim_display = sim.copy()
    sim_display["approval_rate"]          = sim_display["approval_rate"].map(fmt_pct)
    sim_display["portfolio_default_rate"] = sim_display["portfolio_default_rate"].map(fmt_pct)
    sim_display["n_approved"]             = sim_display["n_approved"].map(lambda v: f"{int(v):,}")
    sim_display["expected_defaults"]      = sim_display["expected_defaults"].map(lambda v: f"{v:,.0f}")
    sim_display["expected_loss"]          = sim_display["expected_loss"].map(fmt_currency)
    st.dataframe(sim_display, width="stretch")
    st.caption(
        "Conservative = approve only LOW-band applicants. Moderate = approve "
        "LOW + MEDIUM, pushing a larger book through auto-approval at the cost "
        "of a higher realised default rate."
    )


# --- Page 4: Fairness Report --------------------------------------------------


def _fairness_group_charts(sub: pd.DataFrame, attribute: str) -> None:
    approval_fig = px.bar(
        sub, x="group", y="approval_rate", text_auto=".1%",
        color="approval_rate", color_continuous_scale="Blues",
        title=f"Approval rate by {attribute}",
    )
    approval_fig.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Approval rate",
        xaxis_title="",
        coloraxis_showscale=False,
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(approval_fig, width="stretch")

    long = sub.melt(
        id_vars="group",
        value_vars=["FPR", "FNR"],
        var_name="metric",
        value_name="rate",
    )
    err_fig = px.bar(
        long, x="group", y="rate", color="metric", barmode="group",
        text_auto=".1%",
        color_discrete_map={"FPR": DECISION_COLOR["APPROVE"], "FNR": DECISION_COLOR["REJECT"]},
        title=f"Error rates by {attribute} (FPR = good rejected, FNR = defaulter approved)",
    )
    err_fig.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Rate",
        xaxis_title="",
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="",
    )
    st.plotly_chart(err_fig, width="stretch")


def page_fairness() -> None:
    st.title("Fairness Report")
    st.caption(
        "Group-wise approval and error rates on the held-out test set. "
        "A disparity here is diagnostic — it does not, by itself, prove "
        "discrimination, and it does not substitute for a legal/compliance review."
    )

    df = load_csv("fairness_metrics")
    if df is None:
        warn_missing("fairness_metrics")
        return

    gender_df = df[df["attribute"] == "gender"].sort_values("group")
    edu_df    = df[df["attribute"] == "education"].sort_values("approval_rate", ascending=False)

    st.subheader("Gender")
    if gender_df.empty:
        st.info("No gender rows in fairness_metrics.csv.")
    else:
        _fairness_group_charts(gender_df, "gender")

    st.divider()

    st.subheader("Education")
    if edu_df.empty:
        st.info("No education rows in fairness_metrics.csv.")
    else:
        _fairness_group_charts(edu_df, "education")

    st.divider()

    st.subheader("Summary")
    st.markdown(
        """
**Key findings**
- Approval-rate spread exists across both gender and education groups, driven largely by
  differences in underlying default rates and in the distribution of `EXT_SOURCE_*` scores.
- Error rates (FPR, FNR) are not equal across groups — equalized-odds parity does not hold.
- Calibration is broadly consistent across gender, so the same predicted probability implies
  a similar realised default rate regardless of group (see `notebooks/06_fairness.ipynb`).

**Caveats**
- A disparity can reflect genuine risk differences, data gaps, or historical inequities —
  these causes are not separable from a descriptive audit alone.
- Small groups (e.g. `Academic degree`, `CODE_GENDER = XNA`) are filtered (`n < 100`) to
  avoid unstable rates.
- The approval definition used here matches `test_scores.csv` — i.e. APPROVE = not HIGH risk.
  A stricter `approve = LOW only` policy changes these numbers.

**Diagnostic, not a compliance judgment.** Any production use of this model should
include a legal/compliance review, ongoing stratified monitoring (approval rate, default
rate, FPR/FNR, mean score by group), and a less-discriminatory-alternative search
when disparities cross regulator-defined thresholds.
        """
    )


# --- Router -------------------------------------------------------------------


PAGES = {
    "Model Overview":      page_overview,
    "Score Applicant":     page_score,
    "Portfolio Analytics": page_portfolio,
    "Fairness Report":     page_fairness,
}


def main() -> None:
    st.set_page_config(
        page_title="Credit Risk Decision Engine",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.sidebar.title("Credit Risk Decision Engine")
    st.sidebar.caption("Analyst dashboard")
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown(
        f"""
        **Model:** XGBoost + isotonic
        **Thresholds:** LOW < `{T_LOW:.3f}`, HIGH > `{T_HIGH:.3f}`
        **Artifacts root:** `{REPO_ROOT.name}/`
        """
    )
    PAGES[choice]()


if __name__ == "__main__":
    main()
