## Data Prep
- Weekly granularity enforced; `week` parsed to datetime and sorted.
- Zeros allowed for spends; `log1p` used where needed to stabilize.
- Controls kept as-is: `average_price`, `promotions`, CRM metrics and seasonality/trend (via `DateFeatures`).

## Modeling Approach
- Mediation-aware two-stage MMM.
  - Stage-1: Predict `google_spend` from social channels with adstocked + log1p features, controls, and date features using ElasticNet with blocked `TimeSeriesSplit`.
  - Stage-2: Predict `revenue` from predicted Google (`g_hat`), price/promo, controls, and date features using ElasticNet; optional non-linear HGB with residualized social.
- 20% final holdout after sorting by `week` for end evaluation.

## Causal Framing & Leakage
- DAG: Social → Google (mediator) → Revenue, with potential Social → Revenue.
- Avoid post-treatment bias by using predicted Google (not realized) in Stage-2.
- Time-respecting CV avoids look-ahead; controls reduce omitted-variable risks.

## Diagnostics
- Metrics (train/holdout): R2, RMSE, MAPE.
- Plots: Actual vs Predicted on train and test; coefficient table for ElasticNet.
- Optional: boosting with residualized social to probe non-linear/direct effects.
  - Notebook cells: Stage-1 grid (Cell 5), Stage-2 results (Cell 6), optional boosting (Cell 7), sensitivity (Cell 8).

## Insights & Recs
- Use coefficients and mediated path contributions to guide spend: social that most lifts `g_hat` tends to amplify search and downstream revenue.
- Price elasticity: inspect sensitivity plots and consider guardrails for pricing/promo.
- Next: tune λ per channel, add uncertainty, extend to optimizer for budget allocation.

