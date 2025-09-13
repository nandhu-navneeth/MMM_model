import argparse, json, os, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from src.seed_utils import set_seeds
from src.mmm_utils import DateFeatures, adstock_geometric, metrics_dict


def run(data_path, output_dir):
    set_seeds(42)
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df = df.sort_values("week").reset_index(drop=True)

    target_col   = "revenue"
    mediator_col = "google_spend"
    social_cols  = ["facebook_spend","tiktok_spend","instagram_spend","snapchat_spend"]
    controls     = ["social_followers","emails_send","sms_send"]
    price_col    = "average_price"
    promo_col    = "promotions"

    X_dates = DateFeatures("week").transform(df)
    n = len(df); test_size = max(1, int(math.ceil(0.20 * n))); split_ix = n - test_size

    # ---- Stage 1: select adstock λ by OOF R2
    tscv = TimeSeriesSplit(n_splits=5)
    lam_grid = [0.3, 0.5, 0.6, 0.7, 0.9]
    best = (-1e9, None, None)
    for lam in lam_grid:
        X_ad = pd.DataFrame({f"{c}_adstock_log1p": np.log1p(adstock_geometric(df[c].fillna(0), lam)) for c in social_cols})
        X1 = pd.concat([X_ad, X_dates, df[controls].fillna(0)], axis=1)
        y1 = np.log1p(df[mediator_col].fillna(0))
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=np.logspace(-4,1,40),
                                  cv=tscv, max_iter=20000, random_state=42))
        ])
        oof = np.zeros_like(y1, dtype=float)
        for tr, te in tscv.split(X1):
            pipe.fit(X1.iloc[tr], y1.iloc[tr])
            oof[te] = pipe.predict(X1.iloc[te])
        oof_r2 = np.corrcoef(y1, oof)[0,1]**2 if not np.isnan(oof).any() else -1e9
        if oof_r2 > best[0]: best = (oof_r2, lam, pipe.fit(X1, y1))
    lam_star, model1 = best[1], best[2]

    df["g_log1p_hat"] = model1.predict(pd.concat([
        pd.DataFrame({f"{c}_adstock_log1p": np.log1p(adstock_geometric(df[c].fillna(0), lam_star)) for c in social_cols}),
        X_dates, df[controls].fillna(0)
    ], axis=1))
    df["g_hat"] = np.expm1(df["g_log1p_hat"])

    # ---- Stage 2 (ElasticNet)
    def make_stage2_features(dd):
        X = pd.DataFrame({
            "log1p_avg_price": np.log1p(dd[price_col].fillna(0)),
            "log1p_g_hat":     np.log1p(dd["g_hat"].fillna(0)),
        })
        return pd.concat([X, dd[[promo_col]+controls].fillna(0), X_dates], axis=1)

    X2 = make_stage2_features(df)
    y2 = np.log1p(df[target_col].clip(lower=1e-6))
    X2_tr, X2_te = X2.iloc[:split_ix], X2.iloc[split_ix:]
    y2_tr, y2_te = y2.iloc[:split_ix], y2.iloc[split_ix:]

    enet = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=np.logspace(-4,1,40),
                              cv=TimeSeriesSplit(n_splits=5), max_iter=20000, random_state=42))
    ])
    enet.fit(X2_tr, y2_tr)
    pred_tr = np.expm1(enet.predict(X2_tr)); pred_te = np.expm1(enet.predict(X2_te))
    act_tr = np.expm1(y2_tr);                act_te = np.expm1(y2_te)

    m_tr = metrics_dict(act_tr, pred_tr); m_te = metrics_dict(act_te, pred_te)
    with open(os.path.join(output_dir, "metrics_elasticnet.json"), "w") as f: json.dump({"train":m_tr,"test":m_te}, f, indent=2)

    # Plots
    plt.figure(); plt.plot(df["week"].iloc[:split_ix], act_tr, label="Actual (train)")
    plt.plot(df["week"].iloc[:split_ix], pred_tr, label="Pred (train)"); plt.legend(); plt.title("Train"); plt.savefig(os.path.join(output_dir,"train_curve.png"), dpi=160); plt.close()

    plt.figure(); plt.plot(df["week"].iloc[split_ix:], act_te, label="Actual (test)")
    plt.plot(df["week"].iloc[split_ix:], pred_te, label="Pred (test)"); plt.legend(); plt.title("Test"); plt.savefig(os.path.join(output_dir,"test_curve.png"), dpi=160); plt.close()

    print("Saved metrics & plots to:", output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="out")
    args = p.parse_args()
    run(args.data, args.out)

