from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .metrics import rmse, mape, r2


@dataclass
class CVConfig:
    n_splits: int = 5
    test_size: int = 8  # weeks
    gap: int = 0
    random_state: int = 42


def expanding_time_splits(n: int, test_size: int, n_splits: int, gap: int = 0):
    """Yield expanding window splits with fixed test_size and optional gap."""
    start = 0
    for i in range(n_splits):
        train_end = n - (n_splits - i) * test_size - gap
        if train_end <= start:
            continue
        test_start = train_end + gap
        test_end = min(test_start + test_size, n)
        if test_start >= n or test_start >= test_end:
            break
        yield np.arange(start, train_end), np.arange(test_start, test_end)


def build_enet_pipeline(numeric_features: List[str], alpha: float, l1_ratio: float, random_state: int = 42) -> Pipeline:
    pre = ColumnTransformer([
        ("num", RobustScaler(with_centering=True, with_scaling=True), numeric_features)
    ], remainder="drop")
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=10000)
    pipe = Pipeline([
        ("prep", pre),
        ("model", model),
    ])
    return pipe


def cv_grid_search(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    grid: Dict[str, List],
    cv_conf: CVConfig,
):
    y = df[target_col].to_numpy()
    X = df[feature_cols]

    results = []
    n = len(df)
    for alpha in grid.get("alpha", [0.1]):
        for l1_ratio in grid.get("l1_ratio", [0.5]):
            y_preds_all = np.full(n, np.nan)
            metrics = []
            for tr_idx, te_idx in expanding_time_splits(n, cv_conf.test_size, cv_conf.n_splits, cv_conf.gap):
                pipe = build_enet_pipeline(feature_cols, alpha, l1_ratio, random_state=cv_conf.random_state)
                pipe.fit(X.iloc[tr_idx], y[tr_idx])
                preds = pipe.predict(X.iloc[te_idx])
                y_preds_all[te_idx] = preds
                metrics.append({
                    "rmse": rmse(y[te_idx], preds),
                    "mape": mape(y[te_idx], preds),
                    "r2": r2(y[te_idx], preds),
                })
            mean_metrics = {
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "rmse": float(np.nanmean([m["rmse"] for m in metrics])),
                "mape": float(np.nanmean([m["mape"] for m in metrics])),
                "r2": float(np.nanmean([m["r2"] for m in metrics])),
            }
            results.append(mean_metrics)

    # pick best by RMSE
    results = sorted(results, key=lambda d: d["rmse"])  # ascending
    best = results[0]
    return best, results


