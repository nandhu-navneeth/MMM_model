from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from mmm.data_utils import ColumnMap, load_weekly_csv, fill_and_impute, add_calendar_features, check_required_columns
from mmm.transforms import batch_adstock_saturation
from mmm.mediation import fit_two_stage_mediation
from mmm.modeling import CVConfig
from mmm.metrics import rmse, mape, r2


def build_features(df: pd.DataFrame, cmap: ColumnMap, adstock_grid: Dict) -> pd.DataFrame:
    df = add_calendar_features(df, cmap.date, fourier_k=3)
    # Apply adstock + saturation to spends
    spend_cols = [cmap.google, *cmap.social]
    df = batch_adstock_saturation(df, spend_cols, adstock_grid, prefix="x")
    # Direct levers can use log1p to stabilize scale
    for col in cmap.direct:
        if col in df.columns:
            df[f"x_{col}"] = np.log1p(df[col].fillna(0.0))
    # Controls may pass through as is
    for col in cmap.controls:
        if col in df.columns and col not in [cmap.date, cmap.target]:
            df[f"c_{col}"] = df[col]
    # Calendar feats already added
    return df


def main():
    parser = argparse.ArgumentParser(description="Train mediation‑aware MMM")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with weekly data")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON column map config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=str, default="outputs/report.txt")
    args = parser.parse_args()

    cmap = ColumnMap.from_config(args.config)
    df = load_weekly_csv(args.data, cmap.date, enforce_weekly=True)
    check_required_columns(df, cmap)
    df = fill_and_impute(df, cmap)

    # Default adstock/saturation grid per channel (tunable)
    adstock_grid = {
        cmap.google: {"decay": 0.5, "sat_type": "log", "sat_param": 1000.0},
        **{ch: {"decay": 0.5, "sat_type": "log", "sat_param": 1000.0} for ch in cmap.social},
    }
    df = build_features(df, cmap, adstock_grid)

    # Define features
    mediator_features: List[str] = []
    for ch in cmap.social:
        if ch in df.columns:
            mediator_features.append(f"x_{ch}")
    # option: include direct levers if you believe they also stimulate search
    # mediator_features += [f"x_{c}" for c in cmap.direct if f"x_{c}" in df.columns]
    # Seasonality/controls for mediator
    mediator_features += [c for c in df.columns if c.startswith("fourier_") or c == "trend"]

    outcome_features: List[str] = []
    # instrumented mediator will be added later; do not include original mediator feature
    for ch in cmap.social:
        if ch in df.columns:
            outcome_features.append(f"x_{ch}")
    outcome_features += [f"x_{c}" for c in cmap.direct if f"x_{c}" in df.columns]
    outcome_features += [f"c_{c}" for c in cmap.controls if f"c_{c}" in df.columns]
    outcome_features += [c for c in df.columns if c.startswith("fourier_") or c == "trend"]

    # Cross‑validation config
    n_weeks = len(df)
    cv_conf = CVConfig(n_splits=min(5, max(2, n_weeks // 12)), test_size=max(6, n_weeks // 10), gap=0, random_state=args.seed)

    # Hyperparameter grids
    grid_mediator = {"alpha": [0.01, 0.05, 0.1, 0.5], "l1_ratio": [0.1, 0.5, 0.9]}
    grid_outcome = {"alpha": [0.01, 0.05, 0.1, 0.5], "l1_ratio": [0.1, 0.5, 0.9]}

    # Fit two‑stage mediation
    res = fit_two_stage_mediation(
        df,
        mediator_col=cmap.google,
        outcome_col=cmap.target,
        mediator_features=mediator_features,
        outcome_features=outcome_features,
        cv_conf=cv_conf,
        grid_mediator=grid_mediator,
        grid_outcome=grid_outcome,
    )

    # In‑sample diagnostics summary
    df = df.copy()
    df["mediator_hat"] = res.mediator_model.predict(df[res.mediator_features])
    y_true = df[cmap.target].to_numpy()
    y_pred = res.outcome_model.predict(df[res.outcome_features])
    summary = {
        "rmse_insample": rmse(y_true, y_pred),
        "mape_insample": mape(y_true, y_pred),
        "r2_insample": r2(y_true, y_pred),
        "mediator_cv_best": res.mediator_best,
        "outcome_cv_best": res.outcome_best,
        "n_weeks": int(n_weeks),
    }

    # CV diagnostics (out‑of‑sample)
    from mmm.mediation import cross_validate_two_stage
    cv_diag = cross_validate_two_stage(
        df,
        mediator_col=cmap.google,
        outcome_col=cmap.target,
        mediator_features=res.mediator_features,
        base_outcome_features=[c for c in res.outcome_features if c != "mediator_hat"],
        cv_conf=cv_conf,
        alpha_med=res.mediator_best["alpha"],
        l1_med=res.mediator_best["l1_ratio"],
        alpha_out=res.outcome_best["alpha"],
        l1_out=res.outcome_best["l1_ratio"],
    )

    cv_rmse = float(np.nanmean([m["rmse"] for m in cv_diag["metrics"]]))
    cv_mape = float(np.nanmean([m["mape"] for m in cv_diag["metrics"]]))
    cv_r2 = float(np.nanmean([m["r2"] for m in cv_diag["metrics"]]))

    # Save a simple report
    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("Mediation MMM Report\n")
        f.write("====================\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\nCV (blocked, expanding) metrics:\n")
        f.write(f"  rmse: {cv_rmse}\n")
        f.write(f"  mape: {cv_mape}\n")
        f.write(f"  r2: {cv_r2}\n")
        f.write("\nFeatures (mediator):\n")
        for c in res.mediator_features:
            f.write(f"  - {c}\n")
        f.write("\nFeatures (outcome):\n")
        for c in res.outcome_features:
            f.write(f"  - {c}\n")
        # Coefficients (on scaled features)
        try:
            coef_med = res.mediator_model.named_steps["model"].coef_
            f.write("\nMediator coefficients (scaled feature space):\n")
            for name, coef in zip(res.mediator_features, coef_med):
                f.write(f"  {name}: {coef}\n")
        except Exception:
            pass
        try:
            coef_out = res.outcome_model.named_steps["model"].coef_
            f.write("\nOutcome coefficients (scaled feature space):\n")
            for name, coef in zip(res.outcome_features, coef_out):
                f.write(f"  {name}: {coef}\n")
        except Exception:
            pass

    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
