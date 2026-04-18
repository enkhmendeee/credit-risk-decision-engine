# Model Card — Credit Risk Decision Engine

A calibrated gradient-boosted classifier that predicts the probability a loan applicant will default on a Home Credit consumer loan, together with a risk-band policy and SHAP-based reason codes. This card documents the model's design, performance, fairness properties, and intended operating envelope.

- **Model name:** `xgboost_calibrated`
- **Artifact path:** `models/xgboost_calibrated.pkl`
- **Version:** 0.1.0
- **Card date:** 2026-04-18
- **Owner:** Credit Risk Decision Engine project
- **License:** internal / research

---

## 1. Model details

### 1.1 Model type

Isotonic-calibrated XGBoost binary classifier for the task of predicting `TARGET ∈ {0, 1}`, where `1` marks an applicant who experienced payment difficulties on the Home Credit consumer loan.

- Base learner: `xgboost.XGBClassifier`
  - `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`
  - `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`
  - `scale_pos_weight=11` to compensate for the ~8% minority class
- Calibrator: `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`
  - Monotone post-hoc transform on the raw XGBoost probability
  - Implemented as `IsotonicCalibratedModel` wrapper (see `src/train.py`) which exposes both the calibrated probability and the underlying tree model for SHAP
- Training split: 80/20 stratified on `TARGET`, `random_state=42`
- Calibration data: the held-out test set itself (noted as a limitation below)

### 1.2 Training data

- **Dataset:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
  - `application_train.csv` — primary applicant records
  - `bureau.csv`, `bureau_balance.csv` — external bureau history (used in EDA; not joined into the engineered training frame in this version)
- **Size:** ~307,511 applicants after feature engineering
- **Base rate:** ~8.07% default
- **Target:** binary `TARGET` (1 = payment difficulties observed, 0 = not observed)
- **Demographic composition (test set, 61,503 rows):**
  - `CODE_GENDER`: 40,561 F · 20,940 M · 2 XNA (dropped by min-group filter)
  - `NAME_EDUCATION_TYPE`: Secondary / secondary special 43,623 · Higher education 15,061 · Incomplete higher 1,988 · Lower secondary 791 · Academic degree 40 (dropped by min-group filter)

### 1.3 Features

101 features fed to the model after the pipeline in `src/features.py`:

- **Raw applicant attributes** — `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`, etc.
- **Engineered age / employment** — `AGE_YEARS` (from `DAYS_BIRTH`), `YEARS_EMPLOYED`, `DAYS_EMPLOYED_anomaly` flag
- **Financial ratios** — `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`, `EMPLOYED_LIFE_RATIO`, `INCOME_PER_PERSON`
- **External bureau score aggregates** — `EXT_SOURCE_1/2/3`, `EXT_SOURCE_MEAN`, `EXT_SOURCE_MIN`, `EXT_SOURCE_WEIGHTED`
- **Document and social-circle signals** — `DOCUMENT_FLAGS_SUM`, `SOCIAL_CIRCLE_DEFAULT_RATE`
- **Categorical fields** — `CODE_GENDER`, `NAME_EDUCATION_TYPE`, `NAME_INCOME_TYPE`, `NAME_CONTRACT_TYPE`, `ORGANIZATION_TYPE`, etc., label-encoded per column
- **Missingness indicators** — `<column>_missing` binary flags for columns with >20% missing (notably `EXT_SOURCE_1_missing`)

### 1.4 Risk-policy layer

Applicants are bucketed into three risk bands by calibrated probability, with thresholds defined in `configs/config.yaml`:

| Band | Rule | Operational meaning |
| --- | --- | --- |
| LOW | `p < 0.0649` | auto-approve |
| MEDIUM | `0.0649 ≤ p ≤ 0.1388` | route to human review |
| HIGH | `p > 0.1388` | auto-decline, with adverse-action reasons |

Thresholds sit at roughly the 60th and 85th percentiles of the calibrated score distribution on the test set.

---

## 2. Intended use

### 2.1 Primary intended uses

- **Tier-1 prescreening** of consumer-loan applicants. The model produces a risk score, a band (LOW / MEDIUM / HIGH), and — for HIGH cases — a short list of SHAP-derived reason codes suitable for adverse-action notices.
- **Portfolio simulation and threshold tuning.** Risk teams can use `simulate_portfolio` and `threshold_analysis` (in `src/policy.py`) to compare approval rates, portfolio default rates, and expected losses under different policies.
- **Analyst review of individual applicants.** The Streamlit dashboard (`dashboard/app.py`) exposes the full scoring path — KPIs, calibration, feature importance, per-applicant SHAP waterfalls, portfolio analytics, and a fairness report.
- **Teaching / reference implementation** of a full ML system (feature pipeline, MLflow-tracked training, calibration, explainability, serving, fairness audit) on a public dataset.

### 2.2 Out-of-scope uses

The model should **not** be used:

- As the **sole decision-maker** on any credit application. MEDIUM cases require human review; HIGH-band declines require the adverse-action reason codes to be checked against the applicant's provided data and against local regulation before being sent.
- For **products or populations materially different from Home Credit's training distribution** — for example, business lending, mortgages, different geographies, or significantly different income bands. Performance is unwarranted outside the training distribution.
- To **set interest rates or compute regulatory capital** without an independent recalibration on a dedicated holdout and local validation.
- To produce **causal inferences** about demographic groups. The fairness numbers in this card are descriptive, not causal.
- For any use that is prohibited by local fair-lending, data-protection, or automated-decision-making regulation (for example, fully automated rejection in jurisdictions covered by GDPR Article 22 without the required human-in-the-loop and explanation obligations).

---

## 3. Performance metrics

All numbers below are on the 20% stratified test split (n = 61,503).

### 3.1 Headline metrics

| Metric | Calibrated XGBoost | Raw XGBoost (for contrast) |
| --- | --- | --- |
| AUC-ROC | **0.7712** | 0.7712 (ranking unchanged by monotone calibration) |
| PR-AUC | **0.2496** | 0.2496 |
| KS statistic | **0.4045** | 0.4045 |
| Brier score | **0.0669** | 0.1751 |
| Brier skill score | **0.0989** | −1.3597 |
| Reliability (calibration error) | **0.000000** | 0.107856 |
| Resolution | 0.007380 | 0.006626 |

Isotonic calibration reduces Brier score by ~61.8% without changing the model's ranking ability.

### 3.2 Per-band performance

Distribution of applicants and realised default rates on the test set:

| Band | Share of applicants | Realised default rate |
| --- | --- | --- |
| LOW | 51.2% (31,509) | **2.64%** |
| MEDIUM | 29.0% (17,850) | 8.18% |
| HIGH | 19.7% (12,144) | **22.0%** |

Key risk separation:

- **Approved (not HIGH)** — 80.3% of applicants, 4.65% realised default rate
- **Rejected (HIGH)** — 19.7% of applicants, ~22.7% realised default rate
- ~10× risk separation between approved and rejected populations at the configured thresholds

### 3.3 Portfolio simulation

From `src.policy.simulate_portfolio` on the test set:

| Policy | Approval rate | Portfolio default rate (among approved) |
| --- | --- | --- |
| Conservative (LOW only) | **51.2%** | **2.64%** |
| Moderate (LOW + MEDIUM) | **80.3%** | **4.65%** |

Expected loss is computed as the sum of `AMT_CREDIT` over approved applicants who actually defaulted; absolute values depend on the test-set loan sizes and should be treated as relative comparisons, not dollar forecasts.

---

## 4. Fairness analysis summary

Fairness is assessed descriptively on the test set across two sensitive attributes, using `approved = band ≠ HIGH` to define the operational approval decision. Groups with n < 100 are filtered. Full breakdowns live in `data/fairness_metrics.csv` and `notebooks/06_fairness.ipynb`.

### 4.1 Gender (`CODE_GENDER`)

| Group | n (good) | n (bad) | Approval rate | FPR (good rejected) | FNR (defaulter approved) |
| --- | --- | --- | --- | --- | --- |
| F | 37,725 | 2,836 | **84.3%** | 13.4% | 53.4% |
| M | 18,811 | 2,129 | **72.4%** | 23.6% | 36.6% |

- Approval-rate gap F vs M: **11.9 pp** — flagged (>10 pp threshold commonly used in disparate-impact screening).
- Men have a higher underlying default rate and receive higher predicted probabilities on average, which drives the lower approval rate.
- FPR is higher for men (more good men rejected); FNR is higher for women (more female defaulters approved). The model trades off errors differently across groups.

### 4.2 Education (`NAME_EDUCATION_TYPE`)

| Group | n (good) | n (bad) | Approval rate | FPR | FNR |
| --- | --- | --- | --- | --- | --- |
| Higher education | 14,278 | 783 | **90.5%** | 8.0% | 63.0% |
| Incomplete higher | 1,831 | 157 | 79.4% | 17.7% | 45.9% |
| Secondary / secondary special | 39,681 | 3,942 | 76.9% | 19.7% | 42.9% |
| Lower secondary | 709 | 82 | **71.9%** | 25.0% | 45.1% |

- Approval rate declines monotonically with education level. Gap between `Higher education` and `Lower secondary` is **18.6 pp** — flagged.
- FPR rises from 8.0% (Higher education) to 25.0% (Lower secondary).
- `Academic degree` (n = 40) is below the minimum-group filter and therefore excluded.

### 4.3 Calibration parity

Per-gender reliability diagrams (`notebooks/06_fairness.ipynb`, Section 4) indicate broadly similar calibration — at a given predicted probability, realised default rates for F and M applicants are close, though not identical. This is the most favourable of the three fairness views; equalized-odds parity (Section 4.2) does not hold.

---

## 5. Ethical considerations

- **Fairness is diagnostic, not causal.** A disparity in approval rate or error rate does not by itself establish illegal discrimination. Under US ECOA, a model may use facially neutral features that correlate with protected classes provided the features are predictive, a less-discriminatory alternative has been searched for, and adverse-action notices state specific reasons. This card does not attempt that legal determination.
- **Protected attributes are excluded from reason codes.** `src/score.py` defines `EXCLUDED_FROM_REASONS` (15 demographic / proxy columns) which are never surfaced in adverse-action reasons, even when they rank high on SHAP. This reduces the chance of a reason statement referencing a protected characteristic, but does not eliminate indirect correlation with group membership.
- **Missingness patterns carry meaning.** `EXT_SOURCE_1` is missing for ~56% of applicants and is imputed with the median plus an `EXT_SOURCE_1_missing` flag. If missingness is correlated with a protected group (a plausible hypothesis in emerging-market data), the model is effectively learning a signal from "who has a bureau record" — which should be monitored.
- **Automated adverse decisions face regulatory constraints.** GDPR Article 22, the UK FCA Consumer Duty, CFPB guidance on adverse-action notices, and local fair-lending laws all impose specific requirements on automated credit decisions. Any production deployment must satisfy them.
- **The model is a statistical tool, not an oracle.** A borrower's score reflects patterns in historical Home Credit data; it does not represent the borrower's character, intent, or ability to improve their circumstances.

---

## 6. Limitations

- **Dataset scope.** Home Credit is one sample of one lender's portfolio from one market. Results are not guaranteed to transfer to other products, lenders, or jurisdictions. `bureau.csv` / `bureau_balance.csv` are not currently joined into the engineered frame — doing so is likely to improve both performance and calibration.
- **Calibration on the test set.** The isotonic calibrator is fit on the same data the headline Brier score is computed on. This inflates apparent calibration quality. Production use should calibrate on a dedicated holdout or with cross-validation.
- **Class imbalance.** The 8% default base rate pushes ranking-favourable models to high AUC without necessarily producing actionable probabilities — hence the importance of calibration and per-band monitoring.
- **Label noise and timing.** `TARGET` encodes whether payment difficulties were observed within an observation window; definitions differ across lenders and bureaux. Retraining on a target aligned with the consuming institution's loss definition is strongly recommended.
- **Covariate drift.** Emerging-market credit conditions change fast (macro shocks, product mix, bureau coverage expansion). Performance on applicants drawn from a 2024–2026 distribution is untested.
- **No formal disparate-impact testing.** The fairness numbers here are descriptive statistics. A production audit requires a less-discriminatory-alternative search, counterfactual analysis, and legal review.
- **SHAP values explain the model, not the data-generating process.** A feature with a high SHAP contribution is predictive under this model — it is not necessarily causal.

---

## 7. Recommendations for production use

1. **Do not deploy as a sole decision-maker.** Treat LOW as a recommendation to approve, HIGH as a recommendation to decline with reason codes surfaced to the reviewer, and MEDIUM as mandatory human review.
2. **Recalibrate on a dedicated holdout before launch,** and recalibrate periodically (quarterly at minimum, monthly for fast-moving portfolios).
3. **Instrument stratified monitoring.** Track approval rate, realised default rate, FPR, FNR, and mean predicted probability per month, broken down by:
   - protected attributes (gender, age band, education, region) where legally permissible to collect
   - product type, loan-size band, and channel
   - bureau-coverage status (`EXT_SOURCE_1_missing`)
4. **Monitor covariate drift.** Population Stability Index (PSI) on each input feature, with alerts when PSI > 0.2.
5. **Keep reason codes in scope.** Retain the `EXCLUDED_FROM_REASONS` set; review it whenever a new feature is added to the model.
6. **Audit data quality by group** — missingness of `EXT_SOURCE_*`, label reliability, and feature coverage should be compared across groups before attributing any disparity to the model itself.
7. **Version everything.** Pin the model artifact (`xgboost_calibrated.pkl`), feature list, median-imputation table (`feature_medians.json`), and threshold values together as a single release; the MLflow run ID should be stamped into every scored decision.
8. **Explore mitigation when disparities exceed internal thresholds.** Options include per-group threshold tuning (post-processing), re-weighting of training data, constrained optimisation (equalized-odds post-processing), and feature removal — each with an accuracy trade-off that must be documented.
9. **Engage legal and compliance before go-live.** Jurisdictions differ (US ECOA, EU GDPR Art. 22, UK FCA Consumer Duty, local equivalents). Adverse-action notices, record retention, human-review obligations, and model-risk-management documentation all need explicit sign-off.
10. **Establish a kill-switch and rollback plan.** Every production scoring system should be able to revert to a prior model or to a fully manual underwriting process within minutes.

---

## 8. Contact

Issues, corrections, or fairness concerns should be routed through the project maintainer. This model card will be versioned alongside the model artifact; material changes to training data, features, thresholds, or performance should trigger a new card revision.
