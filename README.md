**Marketing Mix Modeling (MMM) with Mediation**

This repo implements a causal‑aware MMM for weekly data with the explicit assumption that Google spend acts as a mediator between upper‑funnel social/display (Facebook, TikTok, Snapchat) and Revenue. The solution emphasizes interpretability, time‑aware validation, robust preprocessing, and practical recommendations.

**Quick Start**
- Place your CSV in `data/input.csv` (see schema below).
- Run training: `python src/train_mmm.py --data data/input.csv`
- Run sensitivity (price/promo): `python src/run_sensitivity.py --data data/input.csv`
 - Decompose mediated vs direct effects: `python src/run_mediation_decomp.py --data data/input.csv`

**Data Schema (column names are configurable via `--config`)**
- Required: `date`, `revenue`
- Paid media: `facebook_spend`, `tiktok_spend`, `snapchat_spend`, `google_spend`
- Direct response: `email_sends`, `sms_sends`
- Controls: `avg_price`, `promotions`, `followers`

Put your file at `data/input.csv` with at least the required columns. Dates must be weekly (one row per week). Zero‑spend weeks are allowed.

**Modeling Approach (Short Write‑up)**

- Data preparation:
  - Weekly frequency enforced; missing weeks filled with zeros for spend and forward‑fill for slowly moving controls (e.g., followers). Outliers for spends can be optionally winsorized.
  - Seasonality: Fourier series (k=3) on week‑of‑year; linear trend term.
  - Transformations: Geometric adstock for paid media; saturation via a Hill‑like squashing (log1p/hyperbolic alternative) to capture diminishing returns. Zeros handled via `log1p` and adstock inherently accommodates zero periods.
  - Scaling: Robust scaling (median/IQR) in model pipeline to stabilize coefficients.

- Causal framing (Google as mediator):
  - Stage 1 (Mediator model): `google_spend_t` is modeled as a function of upper‑funnel channels (`facebook`, `tiktok`, `snapchat`) with appropriate transforms + seasonality/controls. This estimates search demand stimulation.
  - Stage 2 (Outcome model): `revenue_t` is modeled as a function of predicted mediator (`google_spend_hat`) plus direct paths from upper‑funnel channels and controls. Using the predicted mediator (not realized) helps avoid post‑treatment bias/leakage and aligns with the DAG.
  - DAG intuition: Social → Google → Revenue, with possible direct Social → Revenue. We avoid conditioning on realized Google (a collider for downstream feedback), and use time‑respecting cross‑validation.

- Modeling choice:
  - Elastic Net for interpretability and stability with correlated features. Hyperparameters chosen via blocked, rolling time‑series cross‑validation.
  - Adstock decay and saturation strength are tuned via a small grid in CV. Controls include `avg_price`, `promotions`, `email/sms`, `followers`, seasonality, trend.

- Validation & Diagnostics:
  - Blocked, expanding window CV to prevent look‑ahead. Metrics: RMSE, MAPE, R².
  - Residual analysis: autocorrelation checks, stability of coefficients across folds.
  - Sensitivity: Counterfactual sweeps for `avg_price` and `promotions` to quantify elasticity and promo lift under the fitted model.
  - Mediation decomposition: finite‑difference counterfactual that bumps each social channel by Δ% and separates direct vs mediated lift by holding `mediator_hat` fixed in a second prediction pass.

- Insights & Recommendations (how to use):
  - Decompose effects into direct (social → revenue) and mediated (social → google_hat → revenue). Report elasticities and ROI by channel within current operating range.
  - Flag collinearity risk (spend moving together) and emphasize using instrumented mediator to reduce leakage.
  - Use sensitivity results to propose guardrails: expected revenue changes for ±X% price and promo intensity; recommend budget shifts considering mediated search effects.

**Reproducibility & Craft**
- Deterministic `random_state` everywhere; time‑aware CV; minimal dependencies.
- Code organized into `src/mmm/*` modules and runnable scripts.

**Config**
- See `src/config/default.json` to map your column names if your dataset differs. Override via `--config path/to.json`.

**File Structure**
- `src/mmm/` core library (data prep, transforms, mediation, modeling, metrics)
- `src/train_mmm.py` train pipeline with CV and diagnostics
- `src/run_sensitivity.py` price/promo sensitivity runner
- `data/input.csv` your dataset (not in repo)

**Environment**
- Python >= 3.9, pandas, numpy, scikit‑learn, matplotlib, seaborn.
- Optional: statsmodels (for ACF plots), but not required.

**DAG (Conceptual)**

```
Facebook/TikTok/Snapchat  →  Google Spend (mediator)  →  Revenue
           ↘———————————————(possible direct path)———————————↗
Controls (Price, Promo, Email/SMS, Seasonality, Trend) → Revenue
```

Notes on leakage/back‑door paths:
- We avoid conditioning on realized Google when estimating the outcome to reduce post‑treatment bias; we use `google_hat` predicted from stage 1.
- Back‑door paths like Revenue → Google (via budgets or bidding) are attenuated by instrumenting Google through upper‑funnel and exogenous controls, and by using time‑aware CV.


If you want a notebook instead of scripts, consider converting the scripts into cells; the code is organized to make that straightforward.
