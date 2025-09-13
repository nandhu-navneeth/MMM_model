import argparse, json, os, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from src.seed_utils import set_seeds
from src.mmm_utils import DateFeatures, adstock_geometric, metrics_dict


def run(data_path, output_dir):
    set_seeds(42)
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare
    df = pd.read_csv(data_path)
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df = df.sort_values("week").reset_index(drop=True)

    # Columns per Assessment 2 schema
    target_col   = "revenue"
    mediator_col = "google_spend"
    social_cols  = ["facebook_spend","tiktok_spend","instagram_spend","snapchat_spend"]
    controls     = ["social_followers","emails_send","sms_send"]
    price_col    = "average_price"
    promo_col    = "promotions"

    # Date features and split
    X_dates = DateFeatures("week").transform(df)
    n = len(df)
    test_size = max(1, int(math.ceil(0.20 * n)))
    split_ix = n - test_size

    # ---- Stage 1a: select adstock λ by OOF R2 on log1p(mediator)
    tscv = TimeSeriesSplit(n_splits=5)
    lam_grid = [0.3, 0.5, 0.6, 0.7, 0.9]
    best_r2, lam_star, best_pipe = -1e9, 0.6, None
    y1 = np.log1p(df[mediator_col].fillna(0.0))
    for lam in lam_grid:
        X_ad = pd.DataFrame({
            f"{c}_adstock_log1p": np.log1p(adstock_geometric(df[c].fillna(0.0), lam)) for c in social_cols
        })
        X1 = pd.concat([X_ad, X_dates, df[controls].fillna(0)], axis=1)
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=np.logspace(-4,1,40),
                                  cv=tscv, max_iter=20000, random_state=42))
        ])
        oof = np.zeros_like(y1, dtype=float)
        for tr, te in tscv.split(X1):
            pipe.fit(X1.iloc[tr], y1.iloc[tr])
            oof[te] = pipe.predict(X1.iloc[te])
        oof_r2 = float(np.corrcoef(y1, oof)[0,1]**2) if np.isfinite(oof).all() else -1e9
        if oof_r2 > best_r2:
            best_r2, lam_star, best_pipe = oof_r2, lam, pipe.fit(X1, y1)

    # ---- Stage 1b: hurdle model (any-spend vs intensity)
    X_ad_star = pd.DataFrame({
        f"{c}_adstock_log1p": np.log1p(adstock_geometric(df[c].fillna(0.0), lam_star)) for c in social_cols
    })
    X1_full = pd.concat([X_ad_star, X_dates, df[controls].fillna(0)], axis=1).reset_index(drop=True)
    g = df[mediator_col].fillna(0.0)
    y_bin = (g > 0).astype(int)

    # Logistic: probability any spend
    logit = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
    ])
    logit.fit(X1_full, y_bin)
    p_hat = logit.predict_proba(X1_full)[:, 1]
    auc = float(roc_auc_score(y_bin, p_hat)) if y_bin.nunique() > 1 else float("nan")

    # Intensity: conditional on positive weeks
    pos = y_bin.values == 1
    if pos.sum() >= 10:
        intensity = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=np.logspace(-4,1,40),
                                  cv=TimeSeriesSplit(n_splits=5), max_iter=20000, random_state=42)),
        ])
        intensity.fit(X1_full.iloc[pos], np.log1p(g.iloc[pos]))
        mu_hat = np.expm1(intensity.predict(X1_full))
    else:
        mu_hat = np.full(len(df), g[g>0].median() if (g>0).any() else 0.0)

    # Expected mediator
    df["g_hat"] = p_hat * mu_hat
    df["g_log1p_hat"] = np.log1p(df["g_hat"])

    # ---- Stage 2: build features (add AR lags)
    def build_stage2_features(dd: pd.DataFrame, dates_block: pd.DataFrame) -> pd.DataFrame:
        Xcore = pd.DataFrame({
            "log1p_avg_price": np.log1p(dd[price_col].fillna(0)),
            "log1p_g_hat":     np.log1p(dd["g_hat"].fillna(0)),
        })
        ylog = np.log1p(dd[target_col].clip(lower=1e-6))
        ar = pd.DataFrame({
            "y_log_lag1": ylog.shift(1),
            "y_log_lag2": ylog.shift(2),
        }).fillna(method="bfill")
        X = pd.concat([Xcore, dd[[promo_col] + controls].fillna(0), dates_block, ar], axis=1)
        return X

    X2_full = build_stage2_features(df, X_dates)
    y2_full = np.log1p(df[target_col].clip(lower=1e-6))
    X2_tr, X2_te = X2_full.iloc[:split_ix], X2_full.iloc[split_ix:]
    y2_tr, y2_te = y2_full.iloc[:split_ix], y2_full.iloc[split_ix:]

    # Model A: ElasticNet on log-target
    enet = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=np.logspace(-4,1,40),
                              cv=TimeSeriesSplit(n_splits=5), max_iter=20000, random_state=42)),
    ])
    enet.fit(X2_tr, y2_tr)
    pred_tr_en = np.expm1(enet.predict(X2_tr))
    pred_te_en = np.expm1(enet.predict(X2_te))

    act_tr = np.expm1(y2_tr)
    act_te = np.expm1(y2_te)
    m_tr_en = metrics_dict(act_tr, pred_tr_en)
    m_te_en = metrics_dict(act_te, pred_te_en)

    # Model B: Gradient Boosting on log-target
    hgb = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=800,
        min_samples_leaf=12,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        scoring="neg_mean_squared_error",
    )
    hgb.fit(X2_tr, y2_tr)
    pred_tr_gb = np.expm1(hgb.predict(X2_tr))
    pred_te_gb = np.expm1(hgb.predict(X2_te))
    m_tr_gb = metrics_dict(act_tr, pred_tr_gb)
    m_te_gb = metrics_dict(act_te, pred_te_gb)

    # Choose best model by test RMSE
    best_name = "ENet" if m_te_en["RMSE"] <= m_te_gb["RMSE"] else "HGB"
    best_model = enet if best_name == "ENet" else hgb

    # Save metrics summary
    meta = {
        "lam_star": lam_star,
        "mediator_oof_r2": best_r2,
        "mediator_auc_any_spend": auc,
        "best_model": best_name,
        "metrics_elasticnet": {"train": m_tr_en, "test": m_te_en},
        "metrics_hgb": {"train": m_tr_gb, "test": m_te_gb},
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Coefficients table if ENet
    if best_name == "ENet":
        coef = pd.Series(enet.named_steps["enet"].coef_, index=X2_tr.columns).sort_values(ascending=False)
        coef.to_frame("coef").to_csv(os.path.join(output_dir, "stage2_enet_coefs.csv"))

    # Predictions CSV (train+test)
    pred_df = pd.DataFrame({
        "week": df["week"],
        "actual_revenue": df[target_col],
        "pred_revenue_best": np.r_[pred_tr_en if best_name=="ENet" else pred_tr_gb,
                                     pred_te_en if best_name=="ENet" else pred_te_gb],
        "g_hat": df["g_hat"],
        "average_price": df[price_col],
        "promotions": df[promo_col],
    })
    pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # Plots
    plt.figure()
    plt.plot(df["week"].iloc[:split_ix], act_tr, label="Actual (train)")
    plt.plot(df["week"].iloc[:split_ix], (pred_tr_en if best_name=="ENet" else pred_tr_gb),
             label=f"Pred {best_name} (train)")
    plt.legend(); plt.title("Revenue — Train")
    plt.savefig(os.path.join(output_dir, "train_curve.png"), dpi=160); plt.close()

    plt.figure()
    plt.plot(df["week"].iloc[split_ix:], act_te, label="Actual (test)")
    plt.plot(df["week"].iloc[split_ix:], (pred_te_en if best_name=="ENet" else pred_te_gb),
             label=f"Pred {best_name} (test)")
    plt.legend(); plt.title("Revenue — Test")
    plt.savefig(os.path.join(output_dir, "test_curve.png"), dpi=160); plt.close()

    print("Saved metrics, predictions, and plots to:", output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Assessment2 training script matching notebook logic")
    p.add_argument("--data", required=True, help="Path to 'Assessment 2 - MMM Weekly.csv'")
    p.add_argument("--out", default="out", help="Output directory for artifacts")
    args = p.parse_args()
    run(args.data, args.out)
