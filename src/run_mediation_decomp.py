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
    parser = argparse.ArgumentParser(description="Decompose mediated vs direct effects for social channels")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.1, help="Relative bump in social spend (e.g., 0.1 = +10%)")
    parser.add_argument("--out", type=str, default="outputs/mediation_decomp.csv")
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
    base_pred = res.outcome_model.predict(df[res.outcome_features])
    base_mean = float(np.mean(base_pred))

    rows = []
    for ch in cmap.social:
        if ch not in df.columns:
            continue
        # bump original spend column by delta
        bumped = df.copy()
        bumped[ch] = bumped[ch] * (1.0 + args.delta)
        # recompute features for bumped scenario
        bumped_feats = build_features(bumped[[c for c in bumped.columns if not c.startswith("x_") and not c.startswith("c_")]], cmap, adstock_grid)
        # mediator_hat under bumped
        bumped_feats["mediator_hat"] = res.mediator_model.predict(bumped_feats[res.mediator_features])
        total_pred = res.outcome_model.predict(bumped_feats[res.outcome_features])
        total_mean = float(np.mean(total_pred))

        # direct only: hold mediator_hat at baseline while using bumped direct social features
        direct_only = bumped_feats.copy()
        direct_only["mediator_hat"] = df["mediator_hat"]
        direct_pred = res.outcome_model.predict(direct_only[res.outcome_features])
        direct_mean = float(np.mean(direct_pred))

        total_lift = total_mean - base_mean
        direct_lift = direct_mean - base_mean
        mediated_lift = total_lift - direct_lift

        rows.append({
            "channel": ch,
            "delta": args.delta,
            "total_lift": total_lift,
            "direct_lift": direct_lift,
            "mediated_lift": mediated_lift,
            "mediated_share": (mediated_lift / total_lift) if abs(total_lift) > 1e-12 else np.nan
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved mediation decomposition to {out_path}")


if __name__ == "__main__":
    run()
