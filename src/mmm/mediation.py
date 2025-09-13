from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .modeling import CVConfig, build_enet_pipeline, cv_grid_search
from .metrics import rmse, mape, r2


@dataclass
class MediationResult:
    mediator_best: dict
    outcome_best: dict
    mediator_features: List[str]
    outcome_features: List[str]
    mediator_model: Pipeline
    outcome_model: Pipeline


def fit_two_stage_mediation(
    df: pd.DataFrame,
    mediator_col: str,
    outcome_col: str,
    mediator_features: List[str],
    outcome_features: List[str],
    cv_conf: CVConfig,
    grid_mediator: Dict[str, List],
    grid_outcome: Dict[str, List],
) -> MediationResult:
    # Stage 1: tune and fit mediator model
    best_med, _ = cv_grid_search(df, mediator_col, mediator_features, grid_mediator, cv_conf)
    mediator_pipe = build_enet_pipeline(mediator_features, alpha=best_med["alpha"], l1_ratio=best_med["l1_ratio"], random_state=cv_conf.random_state)
    mediator_pipe.fit(df[mediator_features], df[mediator_col])
    df = df.copy()
    df["mediator_hat"] = mediator_pipe.predict(df[mediator_features])

    # Stage 2: replace mediator with instrumented version
    def _is_mediator_feat(name: str) -> bool:
        return (
            name == mediator_col
            or name == "mediator_hat"
            or name.endswith(f"_{mediator_col}")  # e.g., x_google_spend
        )
    feat2 = ["mediator_hat"] + [f for f in outcome_features if not _is_mediator_feat(f)]
    best_out, _ = cv_grid_search(df, outcome_col, feat2, grid_outcome, cv_conf)
    outcome_pipe = build_enet_pipeline(feat2, alpha=best_out["alpha"], l1_ratio=best_out["l1_ratio"], random_state=cv_conf.random_state)
    outcome_pipe.fit(df[feat2], df[outcome_col])

    return MediationResult(
        mediator_best=best_med,
        outcome_best=best_out,
        mediator_features=mediator_features,
        outcome_features=feat2,
        mediator_model=mediator_pipe,
        outcome_model=outcome_pipe,
    )


def cross_validate_two_stage(
    df: pd.DataFrame,
    mediator_col: str,
    outcome_col: str,
    mediator_features: List[str],
    base_outcome_features: List[str],
    cv_conf: CVConfig,
    alpha_med: float,
    l1_med: float,
    alpha_out: float,
    l1_out: float,
):
    """Blocked expanding CV for two‑stage pipeline. Returns per‑fold metrics and predictions.
    Uses instrumented mediator within each fold to avoid leakage.
    """
    n = len(df)
    y = df[outcome_col].to_numpy()
    preds_all = np.full(n, np.nan)
    metrics = []
    splits = []

    # generate splits consistent with modeling.cv's expanding_time_splits
    from .modeling import expanding_time_splits

    for tr_idx, te_idx in expanding_time_splits(n, cv_conf.test_size, cv_conf.n_splits, cv_conf.gap):
        tr = df.iloc[tr_idx].copy()
        te = df.iloc[te_idx].copy()

        # Stage 1 on train
        med_pipe = build_enet_pipeline(mediator_features, alpha=alpha_med, l1_ratio=l1_med, random_state=cv_conf.random_state)
        med_pipe.fit(tr[mediator_features], tr[mediator_col])
        tr_med_hat = med_pipe.predict(tr[mediator_features])
        te_med_hat = med_pipe.predict(te[mediator_features])

        # Stage 2 on train with instrumented mediator
        def _is_mediator_feat(name: str) -> bool:
            return (
                name == mediator_col
                or name == "mediator_hat"
                or name.endswith(f"_{mediator_col}")
            )
        feat2 = ["mediator_hat"] + [f for f in base_outcome_features if not _is_mediator_feat(f)]
        tr2 = tr.copy(); tr2["mediator_hat"] = tr_med_hat
        te2 = te.copy(); te2["mediator_hat"] = te_med_hat

        out_pipe = build_enet_pipeline(feat2, alpha=alpha_out, l1_ratio=l1_out, random_state=cv_conf.random_state)
        out_pipe.fit(tr2[feat2], tr[outcome_col])
        te_pred = out_pipe.predict(te2[feat2])
        preds_all[te_idx] = te_pred
        m = {"rmse": rmse(y[te_idx], te_pred), "mape": mape(y[te_idx], te_pred), "r2": r2(y[te_idx], te_pred)}
        metrics.append(m)
        splits.append({"train_start": int(tr_idx.min()), "train_end": int(tr_idx.max()), "test_start": int(te_idx.min()), "test_end": int(te_idx.max())})

    return {"metrics": metrics, "preds": preds_all, "splits": splits}
