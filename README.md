[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nandhu-navneeth/MMM_model/blob/main/notebooks/Assessment2_MMM_Mediation.ipynb)

# MMM (Mediation-Aware): Social → Google (Mediator) → Revenue
- Deterministic, time-aware MMM with adstock+saturation, blocked CV, final holdout.
- Stage-1: Predict Google spend from social (mediated path).
- Stage-2: Predict revenue from **predicted** Google + price, promotions, CRM controls, seasonality & trend.
- Optional non-linear Stage-2 (boosting) with residualized social for direct effects.

## Data Schema
- Required: `week`, `revenue`, `google_spend`.
- Social: `facebook_spend`, `tiktok_spend`, `instagram_spend`, `snapchat_spend`.
- CRM/Controls: `social_followers`, `emails_send`, `sms_send`, `average_price`, `promotions`.
- Notes: weekly granularity; dates parsable; zeros allowed.

## Environment & Repro
- Create venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install: `pip install -r requirements.txt`
- CLI run (assessment-2): `python src/train_mmm_assessment2.py --data YOUR.csv --out out/`
- Alternatively (existing CLI): `python src/train_mmm.py --data data/input.csv`
- Notebook: use the Colab badge above and run all cells.
 - Note (Colab/Python 3.12): the notebook uses `numpy>=2.0,<2.3` to avoid binary incompatibilities with system packages; local `requirements.txt` remains pinned for determinism.

## Causal Framing (DAG)
- Assumed DAG: Social → Google → Revenue, with possible Social → Revenue.
- Leakage avoidance: use predicted Google (from Stage-1) in Stage-2, not realized Google.
- Time-respecting validation prevents look-ahead bias.

## Validation Protocol
- Blocked `TimeSeriesSplit` for hyperparameters; 20% final holdout.
- Report: R2, RMSE, MAPE on train and holdout; curves for actual vs predicted.

## Diagnostics
- Plots: train/test curves saved to `out/` by CLI; inline in notebook.
- Coefficients: ElasticNet coefficients table (importance snapshot).
- Residual checks: optional boosting with residualized social for non-linear effects.

## Sensitivity
- Price/promo sensitivity on last week: ±10% price grid, promo ON/OFF.

## Limitations & Next Steps
- Static adstock per-channel; extend to per-channel λ tuning.
- Explore finite impulse response (bounded lags) and Bayesian calibration.
- Add uncertainty bands and budget optimizer on top of the fitted model.

## License
MIT (see `LICENSE`).
