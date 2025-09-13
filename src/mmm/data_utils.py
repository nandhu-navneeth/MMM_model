from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_CONFIG = {
    "date": "date",
    "target": "revenue",
    "google": "google_spend",
    "social": ["facebook_spend", "tiktok_spend", "snapchat_spend"],
    "direct": ["email_sends", "sms_sends"],
    "controls": ["avg_price", "promotions", "followers"],
}


@dataclass
class ColumnMap:
    date: str = "date"
    target: str = "revenue"
    google: str = "google_spend"
    social: Tuple[str, ...] = ("facebook_spend", "tiktok_spend", "snapchat_spend")
    direct: Tuple[str, ...] = ("email_sends", "sms_sends")
    controls: Tuple[str, ...] = ("avg_price", "promotions", "followers")

    @staticmethod
    def from_config(path: Optional[str]) -> "ColumnMap":
        if path is None:
            cfg = DEFAULT_CONFIG
        else:
            cfg = json.loads(Path(path).read_text())
            # Fill missing keys from default for robustness
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)

        return ColumnMap(
            date=cfg["date"],
            target=cfg["target"],
            google=cfg["google"],
            social=tuple(cfg.get("social", [])),
            direct=tuple(cfg.get("direct", [])),
            controls=tuple(cfg.get("controls", [])),
        )


def load_weekly_csv(path: str, date_col: str, enforce_weekly: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    if enforce_weekly:
        df = ensure_weekly_continuity(df, date_col)
    return df.reset_index(drop=True)


def ensure_weekly_continuity(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    idx = pd.date_range(df[date_col].min(), df[date_col].max(), freq="W")
    df = df.set_index(date_col).reindex(idx)
    df.index.name = date_col
    df = df.reset_index()
    return df


def fill_and_impute(
    df: pd.DataFrame,
    cmap: ColumnMap,
    fill_spend_zero: bool = True,
    ffill_controls: bool = True,
) -> pd.DataFrame:
    # Zero‑fill spends and direct volumes by default
    spend_cols = [cmap.google, *cmap.social]
    direct_cols = list(cmap.direct)
    for col in spend_cols + direct_cols:
        if col in df.columns:
            if fill_spend_zero:
                df[col] = df[col].fillna(0.0)

    # Controls: forward fill and then backfill small gaps
    control_cols = list(cmap.controls)
    for col in control_cols:
        if col in df.columns:
            if ffill_controls:
                df[col] = df[col].ffill().bfill()

    # Target: forward fill is not appropriate; leave NaN if missing
    return df


def add_calendar_features(df: pd.DataFrame, date_col: str, fourier_k: int = 3) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["trend"] = np.arange(len(df))
    # Fourier terms for weekly seasonality (annual)
    # 52 weeks approx.
    w = 2 * np.pi * df["weekofyear"] / 52.0
    for k in range(1, fourier_k + 1):
        df[f"fourier_sin_{k}"] = np.sin(k * w)
        df[f"fourier_cos_{k}"] = np.cos(k * w)
    return df


def check_required_columns(df: pd.DataFrame, cmap: ColumnMap) -> None:
    missing = []
    required = [cmap.date, cmap.target, cmap.google]
    for c in required:
        if c not in df.columns:
            missing.append(c)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Warn (don’t fail) if optional columns are missing
    # This allows running with partial datasets.


