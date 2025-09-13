import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


class DateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, date_col="week"): self.date_col = date_col
    def fit(self, X, y=None): return self
    def transform(self, X):
        xx = X.copy()
        xx[self.date_col] = pd.to_datetime(xx[self.date_col], errors="coerce")
        xx["year"] = xx[self.date_col].dt.year
        xx["weekofyear"] = xx[self.date_col].dt.isocalendar().week.astype(int)
        xx["month"] = xx[self.date_col].dt.month
        xx["t"] = np.arange(len(xx))
        return xx[["year","weekofyear","month","t"]]


def adstock_geometric(x, lam=0.6):
    x = np.nan_to_num(np.asarray(x, float), nan=0.0)
    out = np.zeros_like(x, dtype=float)
    for i, v in enumerate(x):
        out[i] = v + (lam * (out[i-1] if i>0 else 0.0))
    return out


def mape(y_true, y_pred, eps=1.0):
    yt = np.maximum(np.asarray(y_true, float), eps)
    yp = np.asarray(y_pred, float)
    return np.mean(np.abs((yt - yp) / yt)) * 100.0


def metrics_dict(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAPE_%": mape(y_true, y_pred),
    }

