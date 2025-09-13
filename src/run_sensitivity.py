from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from mmm.data_utils import ColumnMap, load_weekly_csv, fill_and_impute, add_calendar_features
from mmm.transforms import batch_adstock_saturation
from mmm.mediation import fit_two_stage_mediation
from mmm.modeling import CVConfig


def build_features(df: pd.DataFrame, cmap: ColumnMap, adstock_grid):
    df = add_calendar_features(df, cmap.date, fourier_k=3)
    spend_cols = [cmap.google, *cmap.social]
    df = batch_adstock_saturation(df, spend_cols, adstock_grid, prefix="x")
    for col in cmap.direct:
        if col in df.columns:
            df[f"x_{col}"] = np.log1p(df[col].fillna(0.0))
    for col in cmap.controls:
        if col in df.columns:
            df[f"c_{col}"] = df[col]
    return df


def run():
    parser = argparse.ArgumentParser(description="Price/Promo sensitivity under fitted MMM")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs/sensitivity.csv")
    args = parser.parse_args()

    cmap = ColumnMap.from_config(args.config)
    df = load_weekly_csv(args.data, cmap.date, enforce_weekly=True)
    df = fill_and_impute(df, cmap)

    adstock_grid = {
        cmap.google: {"decay": 0.5, "sat_type": "log", "sat_param": 1000.0},
        **{ch: {"decay": 0.5, "sat_type": "log", "sat_param": 1000.0} for ch in cmap.social},
    }
    df = build_features(df, cmap, adstock_grid)

    mediator_features = [f"x_{ch}" for ch in cmap.social if ch in df.columns]
    mediator_features += [c for c in df.columns if c.startswith("fourier_") or c == "trend"]
    outcome_features: List[str] = []
    outcome_features += [f"x_{ch}" for ch in cmap.social if ch in df.columns]
    outcome_features += [f"x_{c}" for c in cmap.direct if f"x_{c}" in df.columns]
    outcome_features += [f"c_{c}" for c in cmap.controls if f"c_{c}" in df.columns]
    outcome_features += [c for c in df.columns if c.startswith("fourier_") or c == "trend"]

    n_weeks = len(df)
    cv_conf = CVConfig(n_splits=min(5, max(2, n_weeks // 12)), test_size=max(6, n_weeks // 10), gap=0, random_state=args.seed)
    grid = {"alpha": [0.01, 0.05, 0.1, 0.5], "l1_ratio": [0.1, 0.5, 0.9]}
    res = fit_two_stage_mediation(df, cmap.google, cmap.target, mediator_features, outcome_features, cv_conf, grid, grid)

    df = df.copy()
    df["mediator_hat"] = res.mediator_model.predict(df[res.mediator_features])

    # Baseline prediction
    base_pred = res.outcome_model.predict(df[res.outcome_features])
    base_mean = float(np.mean(base_pred))

    # Price sensitivity: sweep ±20% in 5% steps around observed avg_price
    price_col = "c_avg_price"
    promo_col = "c_promotions"
    mod_df = df.copy()

    rows = []
    for pct in [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]:
        if price_col in mod_df.columns:
            tmp = mod_df.copy()
            tmp[price_col] = tmp[price_col] * (1.0 + pct)
            pred = res.outcome_model.predict(tmp[res.outcome_features])
            rows.append({"type": "price", "delta": pct, "mean_pred": float(np.mean(pred)), "lift_vs_base": float(np.mean(pred) - base_mean)})

    # Promotion sensitivity: sweep from 0 to 2x
    for pct in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
        if promo_col in mod_df.columns:
            tmp = mod_df.copy()
            tmp[promo_col] = tmp[promo_col] * pct
            pred = res.outcome_model.predict(tmp[res.outcome_features])
            rows.append({"type": "promo", "delta": pct, "mean_pred": float(np.mean(pred)), "lift_vs_base": float(np.mean(pred) - base_mean)})

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved sensitivity to {out_path}")


if __name__ == "__main__":
    run()
