from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Apply geometric adstock with decay in [0, 1).
    x: array of nonnegative spends/volumes.
    """
    decay = float(np.clip(decay, 0.0, 0.999))
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i in range(len(x)):
        carry = x[i] + decay * carry
        out[i] = carry
    return out


def hill_saturation(x: np.ndarray, alpha: float, beta: float = 1.0) -> np.ndarray:
    """Hill‑type saturation: y = x^beta / (alpha^beta + x^beta).
    alpha controls the half‑saturation level; beta controls curvature.
    """
    x = np.asarray(x, dtype=float)
    alpha = max(1e-9, float(alpha))
    beta = max(1e-9, float(beta))
    return np.power(x, beta) / (np.power(alpha, beta) + np.power(x, beta))


def log_saturation(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Log1p saturation with scale factor to adjust curvature."""
    x = np.asarray(x, dtype=float)
    scale = max(1e-9, float(scale))
    return np.log1p(x / scale)


def adstock_then_saturate(
    x: np.ndarray,
    decay: float = 0.5,
    sat_type: str = "log",
    sat_param: float = 1.0,
    hill_beta: float = 1.0,
) -> np.ndarray:
    a = geometric_adstock(x, decay)
    if sat_type == "hill":
        return hill_saturation(a, alpha=sat_param, beta=hill_beta)
    else:
        return log_saturation(a, scale=sat_param)


def batch_adstock_saturation(
    df: pd.DataFrame,
    columns: Iterable[str],
    params: dict,
    prefix: str,
) -> pd.DataFrame:
    """Apply adstock + saturation to multiple columns with shared params dict:
    params = {col: {"decay": float, "sat_type": "log"|"hill", "sat_param": float, "hill_beta": float}}
    Returns df with new columns named `{prefix}_{col}`.
    """
    df = df.copy()
    for col in columns:
        series = df[col].fillna(0.0).to_numpy()
        p = params.get(col, {})
        decay = float(p.get("decay", 0.5))
        sat_type = str(p.get("sat_type", "log"))
        sat_param = float(p.get("sat_param", 1.0))
        hill_beta = float(p.get("hill_beta", 1.0))
        transformed = adstock_then_saturate(series, decay, sat_type, sat_param, hill_beta)
        df[f"{prefix}_{col}"] = transformed
    return df


