# ml/train_model.py
# ─────────────────────────────────────────────
# Phase 4 — XGBoost + LightGBM Model Training  (v3)
#
# Improvements over v2:
#   • LightGBM challenger: both models searched,
#     winner (by F1) saved per ticker/horizon
#   • n_iter split 25+25 = same search budget as
#     previous 50-iter XGB-only runs
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
from datetime import datetime, timedelta

# Ensure UTF-8 output on Windows (avoids cp1252 errors with box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db_connection import get_connection

from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[WARN] lightgbm not installed — XGBoost only. Run: pip install lightgbm")

os.makedirs("ml/models", exist_ok=True)

# ── Set to True to re-run hyperparameter search ────────────────
TUNE = False

PARAM_GRID = {
    "n_estimators":     [100, 150, 200, 300, 400, 500],
    "max_depth":        [2, 3, 4, 5, 6],
    "learning_rate":    [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
    "subsample":        [0.6, 0.65, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.65, 0.7, 0.8, 0.9],
    "min_child_weight": [1, 2, 3, 4, 5],
    "gamma":            [0, 0.1, 0.2, 0.3, 0.4],
    "reg_alpha":        [0, 0.05, 0.1, 0.3, 0.5],
    "reg_lambda":       [0.75, 1, 1.25, 1.5, 1.75, 2, 2.5],
}

LGBM_PARAM_GRID = {
    "n_estimators":      [100, 200, 300, 500],
    "learning_rate":     [0.01, 0.05, 0.1, 0.15],
    "max_depth":         [3, 5, 7, -1],
    "num_leaves":        [15, 31, 63, 127],
    "min_child_samples": [20, 50, 100],
    "subsample":         [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree":  [0.6, 0.7, 0.8],
    "reg_alpha":         [0, 0.1, 0.5],
    "reg_lambda":        [1, 2, 3],
}

# ── Feature lists ──────────────────────────────────────────────

FEATURES_NO_SENT = [
    "close", "volume",
    "ma_5", "ma_10", "ma_20",
    "volatility_7", "momentum_5",
    "price_lag1", "price_lag2", "price_lag3", "price_lag5",
    "rsi_14", "macd", "macd_signal",
    "bollinger_width", "volume_change", "price_ma_ratio",
    # v3
    "return_1d", "return_5d", "return_10d",
    "day_of_week", "atr_14",
    "high_20d_ratio", "low_20d_ratio",
    # v4 — market context
    "spy_return_1d",
    "qqq_return_1d",
    "vix_level",
    "vix_change",
    # v5 — sector ETFs + Chinese market
    "hsi_return_1d",
    "mchi_return_1d",
    "cny_usd_return",
    "xlk_return_1d",
    "xlf_return_1d",
    "xle_return_1d",
    "xlv_return_1d",
    "xli_return_1d",
    # v6 — earnings
    "earnings_surprise",
    # v7 — long-term regime
    "price_ma252_ratio",
]

FEATURES_WITH_SENT = FEATURES_NO_SENT + [
    "sentiment_today",
    "sentiment_lag1", "sentiment_lag2",
    "sentiment_ma_7", "sentiment_std_7", "sentiment_momentum",
]

# 252d-only extra features — fundamental + macro data
FEATURES_252D_EXTRA = [
    "pe_ratio",
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "treasury_10y",
    "treasury_2y",
    "yield_curve",
]
FEATURES_252D_NO_SENT  = FEATURES_NO_SENT  + FEATURES_252D_EXTRA
FEATURES_252D_WITH_SENT = FEATURES_WITH_SENT + FEATURES_252D_EXTRA

# Prediction horizons: key -> (trading days ahead, train/test cutoff)
# All short horizons use a 90-day rolling cutoff:
#   - test set covers ~60 trading days (March → May) — enough rows to measure F1
#   - model still trains on Jan/Feb 2026, capturing recent market regime
#   - 21d: rows from March to ~May 8 have fully resolvable 21-day-ahead prices (~40 rows)
# 5d uses max_threshold=0 (pure up/down) so labels stay 50/50 regardless of regime.
# 252d keeps a fixed cutoff — market-relative target needs a multi-year test window.
_rolling_cutoff = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")

HORIZONS = {
    "1d":   {"days": 1,   "cutoff": _rolling_cutoff, "balance": True,  "max_threshold": 0.000},
    "5d":   {"days": 5,   "cutoff": _rolling_cutoff, "balance": True,  "max_threshold": 0.000},
    "21d":  {"days": 21,  "cutoff": _rolling_cutoff, "balance": True,  "max_threshold": 0.010},
    "252d": {"days": 252, "cutoff": "2024-01-01",    "balance": True,  "max_threshold": 0.000, "market_relative": True},
}

TRAIN_CUTOFF = _rolling_cutoff

# These tickers have all-UP early training data for 252d (2015-2018 bull market),
# causing CV folds to be single-class. Restrict to 2019+ so CV sees both classes.
RESTRICTED_252D_START = {
    "AAPL": "2019-01-01",
    "MSFT": "2019-01-01",
    "NFLX": "2019-01-01",
}

# 21d degenerate tickers: restrict to post-2020 data so the training distribution
# is closer to the 2025 test period (avoids regime-shift class imbalance).
RESTRICTED_21D_START = {
    "MSFT": "2022-01-01",
    "TSLA": "2022-01-01",
    "AMD":  "2022-01-01",
    "NVDA": "2022-01-01",
}


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_features(conn) -> pd.DataFrame:
    sql = """
        SELECT ticker, date, close, volume,
               ma_5, ma_10, ma_20,
               volatility_7, momentum_5,
               price_lag1, price_lag2,
               rsi_14, macd, macd_signal,
               bollinger_width, volume_change, price_ma_ratio,
               COALESCE(sentiment_today, 0) AS sentiment_today,
               sentiment_lag1, sentiment_lag2,
               sentiment_ma_7, sentiment_std_7, sentiment_momentum,
               COALESCE(return_1d,  0) AS return_1d,
               COALESCE(return_5d,  0) AS return_5d,
               COALESCE(return_10d, 0) AS return_10d,
               COALESCE(day_of_week, 0) AS day_of_week,
               COALESCE(atr_14,     0) AS atr_14,
               COALESCE(high_20d_ratio, 1) AS high_20d_ratio,
               COALESCE(low_20d_ratio,  1) AS low_20d_ratio,
               COALESCE(price_lag3, 0) AS price_lag3,
               COALESCE(price_lag5, 0) AS price_lag5,
               COALESCE(spy_return_1d, 0) AS spy_return_1d,
               COALESCE(qqq_return_1d, 0) AS qqq_return_1d,
               COALESCE(vix_level,     0) AS vix_level,
               COALESCE(vix_change,    0) AS vix_change,
               COALESCE(hsi_return_1d,    0) AS hsi_return_1d,
               COALESCE(mchi_return_1d,   0) AS mchi_return_1d,
               COALESCE(cny_usd_return,   0) AS cny_usd_return,
               COALESCE(xlk_return_1d,    0) AS xlk_return_1d,
               COALESCE(xlf_return_1d,    0) AS xlf_return_1d,
               COALESCE(xle_return_1d,    0) AS xle_return_1d,
               COALESCE(xlv_return_1d,    0) AS xlv_return_1d,
               COALESCE(xli_return_1d,    0) AS xli_return_1d,
               COALESCE(earnings_surprise,  0) AS earnings_surprise,
               COALESCE(price_ma252_ratio,  1) AS price_ma252_ratio,
               COALESCE(pe_ratio,           0) AS pe_ratio,
               COALESCE(revenue_growth_yoy, 0) AS revenue_growth_yoy,
               COALESCE(eps_growth_yoy,     0) AS eps_growth_yoy,
               COALESCE(treasury_10y,       0) AS treasury_10y,
               COALESCE(treasury_2y,        0) AS treasury_2y,
               COALESCE(yield_curve,        0) AS yield_curve,
               target
        FROM features
        ORDER BY ticker, date ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    cols = [
        "ticker", "date", "close", "volume",
        "ma_5", "ma_10", "ma_20",
        "volatility_7", "momentum_5",
        "price_lag1", "price_lag2",
        "rsi_14", "macd", "macd_signal",
        "bollinger_width", "volume_change", "price_ma_ratio",
        "sentiment_today",
        "sentiment_lag1", "sentiment_lag2",
        "sentiment_ma_7", "sentiment_std_7", "sentiment_momentum",
        "return_1d", "return_5d", "return_10d",
        "day_of_week", "atr_14",
        "high_20d_ratio", "low_20d_ratio",
        "price_lag3", "price_lag5",
        "spy_return_1d", "qqq_return_1d",
        "vix_level", "vix_change",
        "hsi_return_1d", "mchi_return_1d", "cny_usd_return",
        "xlk_return_1d", "xlf_return_1d", "xle_return_1d", "xlv_return_1d", "xli_return_1d",
        "earnings_surprise", "price_ma252_ratio",
        "pe_ratio", "revenue_growth_yoy", "eps_growth_yoy",
        "treasury_10y", "treasury_2y", "yield_curve",
        "target"
    ]
    df = pd.DataFrame(rows, columns=cols)

    numeric_cols = [c for c in cols if c not in ("ticker", "date", "target")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = df["target"].astype(int)
    df["date"]   = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────
# Time-based split
# ─────────────────────────────────────────────

def time_split(df: pd.DataFrame, features: list, cutoff: str):
    """Train on data before cutoff, test on data after."""
    train = df[df["date"] <  cutoff]
    test  = df[df["date"] >= cutoff]
    return (
        train[features], test[features],
        train["target"], test["target"]
    )


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'─' * 45}")
    print(f"  {model_name}")
    print(f"{'─' * 45}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Test rows : {len(y_test):,}")
    print(f"{'─' * 45}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_model(X_train, y_train, X_test, y_test,
                model_name: str, save_path: str, saved_params: dict = None,
                use_balance: bool = True, n_cv_splits: int = 3):
    if use_balance:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0
    else:
        scale_pos_weight = 1.0

    tscv = TimeSeriesSplit(n_splits=n_cv_splits)

    # ── XGBoost ───────────────────────────────────────────────────
    if not TUNE:
        # Always skip CV when TUNE=False — use saved params or sensible defaults
        if saved_params:
            print(f"  Using saved params: {saved_params}")
            xgb_model = XGBClassifier(
                **saved_params,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        else:
            print("  No saved params — using defaults (set TUNE=True to search)")
            xgb_model = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
        xgb_model.fit(X_train, y_train, verbose=False)
        xgb_best_params = saved_params or {}
    else:
        print("  XGB search (25 iter)...")
        base = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        search = RandomizedSearchCV(
            base, PARAM_GRID, n_iter=25, scoring="accuracy",
            cv=tscv, n_jobs=-1, random_state=42, verbose=0,
            error_score=0,
        )
        try:
            search.fit(X_train, y_train)
            xgb_cv_score = search.best_score_
            print(f"  XGB CV acc : {xgb_cv_score*100:.1f}%")
            if xgb_cv_score < 0.40:
                print("  [WARN] CV accuracy unreliable (many folds failed) — fitting directly with defaults")
                xgb_model = XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss", random_state=42, verbosity=0,
                )
                xgb_model.fit(X_train, y_train)
                xgb_best_params = {}
            else:
                xgb_best_params = search.best_params_
                print(f"  XGB params : {xgb_best_params}")
                xgb_model = search.best_estimator_
        except ValueError:
            print("  [WARN] All CV folds failed (single-class) — fitting directly with defaults")
            xgb_model = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
            xgb_model.fit(X_train, y_train)
            xgb_best_params = {}

    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, f"{model_name} [XGB]")

    winner_model   = xgb_model
    lgbm_comparison = None

    # ── LightGBM TUNE=False fast path ────────────────────────────
    # Refit LGB with its saved hyperparams (loaded from existing _lgbm.pkl),
    # compare F1 with XGB, winner gets saved to the standard path.
    if LGBM_AVAILABLE and not TUNE:
        lgbm_path = save_path.replace(".pkl", "_lgbm.pkl")
        lgbm_init_params = None
        if os.path.exists(lgbm_path):
            try:
                with open(lgbm_path, "rb") as _f:
                    _old = pickle.load(_f)
                lgbm_init_params = {k: v for k, v in _old.get_params().items()
                                    if k not in ("verbose", "n_jobs")}
                print(f"  LGB refit with saved params")
            except Exception:
                pass

        if lgbm_init_params:
            lgbm_model = LGBMClassifier(**lgbm_init_params, verbose=-1, n_jobs=-1)
        else:
            lgbm_model = LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                is_unbalance=use_balance, random_state=42, verbose=-1, n_jobs=-1,
            )
        lgbm_model.fit(X_train, y_train)
        lgbm_metrics = evaluate_model(lgbm_model, X_test, y_test, f"{model_name} [LGB]")

        with open(lgbm_path, "wb") as _f:
            pickle.dump(lgbm_model, _f)

        if lgbm_metrics["f1"] >= xgb_metrics["f1"]:
            winner_model = lgbm_model
            print(f"  >> LightGBM wins  (F1: LGB {lgbm_metrics['f1']:.3f}  XGB {xgb_metrics['f1']:.3f})")
        else:
            print(f"  >> XGBoost wins   (F1: XGB {xgb_metrics['f1']:.3f}  LGB {lgbm_metrics['f1']:.3f})")

        lgbm_comparison = {
            "xgb_f1": xgb_metrics["f1"],        "lgb_f1":  lgbm_metrics["f1"],
            "xgb_acc": xgb_metrics["accuracy"],  "lgb_acc": lgbm_metrics["accuracy"],
            "winner": "LGB" if lgbm_metrics["f1"] >= xgb_metrics["f1"] else "XGB",
            "lgbm_path": lgbm_path, "xgb_path": save_path,
        }

    # ── LightGBM TUNE=True (full hyperparameter search) ──────────
    # LGB staged to _lgbm path; call apply_lgbm_winners() to promote.
    elif LGBM_AVAILABLE and TUNE:
        print("  LGB search (25 iter)...")
        lgbm_base = LGBMClassifier(
            is_unbalance=use_balance,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        lgbm_search = RandomizedSearchCV(
            lgbm_base, LGBM_PARAM_GRID, n_iter=25, scoring="accuracy",
            cv=tscv, n_jobs=-1, random_state=42, verbose=0,
            error_score=0,
        )
        try:
            lgbm_search.fit(X_train, y_train)
            lgbm_cv_score = lgbm_search.best_score_
            print(f"  LGB CV acc : {lgbm_cv_score*100:.1f}%")
            if lgbm_cv_score < 0.40:
                print("  [WARN] CV accuracy unreliable (many folds failed) — fitting LGB directly with defaults")
                lgbm_model = LGBMClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    is_unbalance=use_balance,
                    random_state=42, verbose=-1, n_jobs=-1,
                )
                lgbm_model.fit(X_train, y_train)
                lgbm_best_params = {}
            else:
                lgbm_best_params = lgbm_search.best_params_
                print(f"  LGB params : {lgbm_best_params}")
                lgbm_model = lgbm_search.best_estimator_
        except ValueError:
            print("  [WARN] All CV folds failed (single-class) — fitting LGB directly with defaults")
            lgbm_model = LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                is_unbalance=use_balance,
                random_state=42, verbose=-1, n_jobs=-1,
            )
            lgbm_model.fit(X_train, y_train)
            lgbm_best_params = {}

        lgbm_metrics = evaluate_model(lgbm_model, X_test, y_test, f"{model_name} [LGB]")

        lgbm_path = save_path.replace(".pkl", "_lgbm.pkl")
        with open(lgbm_path, "wb") as f:
            pickle.dump(lgbm_model, f)

        winner = "LGB" if lgbm_metrics["f1"] >= xgb_metrics["f1"] else "XGB"
        print(f"  >> {'LightGBM' if winner == 'LGB' else 'XGBoost':10s} wins  "
              f"(XGB F1 {xgb_metrics['f1']:.3f}  vs  LGB F1 {lgbm_metrics['f1']:.3f})")
        lgbm_comparison = {
            "xgb_acc": xgb_metrics["accuracy"], "lgb_acc": lgbm_metrics["accuracy"],
            "xgb_f1": xgb_metrics["f1"],        "lgb_f1":  lgbm_metrics["f1"],
            "winner": winner, "lgbm_path": lgbm_path, "xgb_path": save_path,
        }

    with open(save_path, "wb") as f:
        pickle.dump(winner_model, f)
    print(f"  Saved -> {save_path}")
    return winner_model, xgb_metrics, xgb_best_params, lgbm_comparison


# ─────────────────────────────────────────────
# Target computation
# ─────────────────────────────────────────────

def compute_target(df: pd.DataFrame, horizon_key: str, horizon_cfg: dict,
                   tickers: list, days: int, cutoff: str):
    """
    Compute binary target labels for a horizon.
    - 1d              : next-day close > today's close
    - 5d / 21d        : forward return > per-ticker median (capped)
    - 252d (market_relative=True) : stock 252d return > SPY 252d return
    Returns (df_h, ticker_thresholds).
    """
    df_h = df.copy()

    if horizon_key == "1d":
        df_h["target"] = (
            df_h.groupby("ticker")["close"]
            .transform(lambda x: x.shift(-days) > x)
            .astype(float)
        )
        return df_h, {t: 0.0 for t in tickers}

    if horizon_cfg.get("market_relative"):
        spy_dates = (
            df[["date", "spy_return_1d"]]
            .drop_duplicates("date")
            .sort_values("date")
            .copy()
        )
        spy_dates["spy_level"] = (1 + spy_dates["spy_return_1d"].fillna(0)).cumprod()
        spy_dates["spy_fwd"]   = spy_dates["spy_level"].shift(-days) / spy_dates["spy_level"] - 1
        df_h = df_h.merge(spy_dates[["date", "spy_fwd"]], on="date", how="left")

        stock_fwd = df_h.groupby("ticker")["close"].transform(
            lambda x: x.shift(-days) / x - 1
        )
        df_h["target"] = (stock_fwd > df_h["spy_fwd"]).astype(float)
        print(f"\n  Target [{horizon_key}]: stock return > SPY return (market-relative)")
        return df_h, {t: 0.0 for t in tickers}

    # Absolute threshold (5d / 21d)
    ticker_thresholds = {}
    print(f"\n  Per-ticker thresholds [{horizon_key}]:")
    print(f"  {'Ticker':<8} {'Threshold':>10}")
    print(f"  {'-'*20}")
    max_thresh = horizon_cfg["max_threshold"]
    for t in tickers:
        t_train = df[(df["ticker"] == t) & (df["date"] < cutoff)]
        rets = (t_train["close"].shift(-days) / t_train["close"] - 1).dropna()
        thresh = max(float(rets.median()) if len(rets) > 30 else 0.0, 0.0)
        thresh = min(thresh, max_thresh)
        ticker_thresholds[t] = thresh
        print(f"  {t:<8} {thresh*100:>9.2f}%")

    df_h["target"] = (
        df_h.groupby("ticker")["close"]
        .transform(lambda x: (x.shift(-days) / x - 1) > ticker_thresholds.get(x.name, 0.0))
        .astype(float)
    )
    return df_h, ticker_thresholds


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_training(ticker_filter: str = None, horizon_filter: str = None):
    print("=" * 55)
    print("Phase 4 — XGBoost Model Training (multi-horizon)")
    print(f"Horizons : {list(HORIZONS.keys())}")
    print(f"Features no-sent : {len(FEATURES_NO_SENT)}")
    print(f"Features w-sent  : {len(FEATURES_WITH_SENT)}")
    if ticker_filter:
        print(f"Ticker filter    : {ticker_filter.upper()} only")
    if horizon_filter:
        print(f"Horizon filter   : {horizon_filter} only")
    print("=" * 55)

    conn = get_connection()
    df   = load_features(conn)
    conn.close()

    print(f"\nLoaded {len(df):,} rows | "
          f"{df['ticker'].nunique()} tickers | "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    sent_cols = ["sentiment_today", "sentiment_lag1", "sentiment_lag2",
                 "sentiment_ma_7", "sentiment_std_7", "sentiment_momentum"]

    saved_params = {}
    params_path  = "ml/models/best_params.json"
    if os.path.exists(params_path) and not TUNE:
        with open(params_path) as f:
            saved_params = json.load(f)
        print(f"  Loaded saved params for {len(saved_params)} models (TUNE=False)")
    elif TUNE:
        print("  TUNE=True — running full hyperparameter search")

    best_params_log  = {}
    lgbm_comparisons = {}   # {model_key: comparison_dict} — populated during TUNE=True
    tickers = sorted(df["ticker"].unique())
    if ticker_filter:
        tickers = [t for t in tickers if t.upper() == ticker_filter.upper()]
        if not tickers:
            print(f"[ERROR] Ticker '{ticker_filter}' not found in DB.")
            return

    # ── Loop over every horizon ────────────────
    horizons = {k: v for k, v in HORIZONS.items()
                if not horizon_filter or k == horizon_filter}
    if horizon_filter and not horizons:
        print(f"[ERROR] Horizon '{horizon_filter}' not found. Valid: {list(HORIZONS.keys())}")
        return

    for horizon_key, horizon_cfg in horizons.items():
        days     = horizon_cfg["days"]
        cutoff   = horizon_cfg["cutoff"]
        h_suffix = "" if horizon_key == "1d" else f"_{horizon_key}"

        print(f"\n{'#' * 55}")
        print(f"  HORIZON: {horizon_key}  ({days} trading days ahead)")
        print(f"  Train/test split: before / after {cutoff}")
        print(f"{'#' * 55}")

        df_h, ticker_thresholds = compute_target(
            df, horizon_key, horizon_cfg, tickers, days, cutoff
        )

        df_h = df_h.dropna(subset=["target"])
        df_h["target"] = df_h["target"].astype(int)

        all_metrics_no_sent   = []
        all_metrics_with_sent = []

        # Use extended feature lists for 252d (includes fundamentals + macro)
        feat_no_sent  = FEATURES_252D_NO_SENT  if horizon_key == "252d" else FEATURES_NO_SENT
        feat_with_sent = FEATURES_252D_WITH_SENT if horizon_key == "252d" else FEATURES_WITH_SENT

        for ticker in tickers:
            print(f"\n{'━' * 55}")
            print(f"  Training: {ticker}  [{horizon_key}]")
            print(f"{'━' * 55}")

            t_df = df_h[df_h["ticker"] == ticker].copy()

            # ── WITHOUT sentiment ──────────────
            df1 = t_df[feat_no_sent + ["date", "target"]].dropna()
            if len(df1) < 100:
                print(f"  [SKIP] Not enough data for {ticker}")
                continue

            if horizon_key == "252d" and ticker in RESTRICTED_252D_START:
                restrict = pd.Timestamp(RESTRICTED_252D_START[ticker])
                df1 = df1[df1["date"] >= restrict]
                print(f"  [252d] Train window restricted to {RESTRICTED_252D_START[ticker]}+")
            elif horizon_key == "21d" and ticker in RESTRICTED_21D_START:
                restrict = pd.Timestamp(RESTRICTED_21D_START[ticker])
                df1 = df1[df1["date"] >= restrict]
                print(f"  [21d] Train window restricted to {RESTRICTED_21D_START[ticker]}+")

            X1_tr, X1_te, y1_tr, y1_te = time_split(df1, feat_no_sent, cutoff)
            print(f"  No-sent  — train: {len(X1_tr):,}  test: {len(X1_te):,}")
            ns_key = f"{ticker}_no_sent" if horizon_key == "1d" else f"{ticker}_no_sent_{horizon_key}"
            n_splits = 3
            _, m1, p1, cmp1 = train_model(
                X1_tr, y1_tr, X1_te, y1_te,
                f"{ticker} — WITHOUT Sentiment [{horizon_key}]",
                f"ml/models/{ticker}_no_sentiment{h_suffix}.pkl",
                saved_params.get(ns_key) or saved_params.get(f"{ticker}_no_sent"),
                use_balance=horizon_cfg["balance"],
                n_cv_splits=n_splits,
            )
            all_metrics_no_sent.append({**m1, "ticker": ticker})
            best_params_log[ns_key] = p1
            if cmp1:
                lgbm_comparisons[ns_key] = cmp1

            # ── WITH sentiment ─────────────────
            df2 = t_df[feat_with_sent + ["date", "target"]].copy()
            df2[sent_cols] = df2[sent_cols].fillna(0.0)
            df2 = df2.dropna()

            if horizon_key == "252d" and ticker in RESTRICTED_252D_START:
                restrict = pd.Timestamp(RESTRICTED_252D_START[ticker])
                df2 = df2[df2["date"] >= restrict]
            elif horizon_key == "21d" and ticker in RESTRICTED_21D_START:
                restrict = pd.Timestamp(RESTRICTED_21D_START[ticker])
                df2 = df2[df2["date"] >= restrict]

            X2_tr, X2_te, y2_tr, y2_te = time_split(df2, feat_with_sent, cutoff)
            print(f"  With-sent — train: {len(X2_tr):,}  test: {len(X2_te):,}")
            ws_key = f"{ticker}_with_sent" if horizon_key == "1d" else f"{ticker}_with_sent_{horizon_key}"
            _, m2, p2, cmp2 = train_model(
                X2_tr, y2_tr, X2_te, y2_te,
                f"{ticker} — WITH Sentiment [{horizon_key}]",
                f"ml/models/{ticker}_with_sentiment{h_suffix}.pkl",
                saved_params.get(ws_key) or saved_params.get(f"{ticker}_with_sent"),
                use_balance=horizon_cfg["balance"],
                n_cv_splits=n_splits,
            )
            all_metrics_with_sent.append({**m2, "ticker": ticker})
            best_params_log[ws_key] = p2
            if cmp2:
                lgbm_comparisons[ws_key] = cmp2

        # ── Global fallback models for this horizon ──
        print(f"\n{'━' * 55}")
        print(f"  Global fallback models [{horizon_key}]")
        print(f"{'━' * 55}")

        # Global fallback always uses standard features — fundamentals are too
        # ticker-specific to generalise across all 10 companies in one model.
        global_feat_ns = FEATURES_NO_SENT
        global_feat_ws = FEATURES_WITH_SENT

        df1_all = df_h[global_feat_ns + ["date", "target"]].dropna()
        X1_tr, X1_te, y1_tr, y1_te = time_split(df1_all, global_feat_ns, cutoff)
        gns_key = "global_no_sent" if horizon_key == "1d" else f"global_no_sent_{horizon_key}"
        _, _, gp1, gcmp1 = train_model(X1_tr, y1_tr, X1_te, y1_te,
                    f"Global — WITHOUT Sentiment [{horizon_key}]",
                    f"ml/models/model_no_sentiment{h_suffix}.pkl",
                    saved_params.get(gns_key) or saved_params.get("global_no_sent"),
                    use_balance=horizon_cfg["balance"],
                    n_cv_splits=3)
        best_params_log[gns_key] = gp1
        if gcmp1:
            lgbm_comparisons[gns_key] = gcmp1

        df2_all = df_h[global_feat_ws + ["date", "target"]].copy()
        df2_all[sent_cols] = df2_all[sent_cols].fillna(0.0)
        df2_all = df2_all.dropna()
        X2_tr, X2_te, y2_tr, y2_te = time_split(df2_all, global_feat_ws, cutoff)
        gws_key = "global_with_sent" if horizon_key == "1d" else f"global_with_sent_{horizon_key}"
        _, _, gp2, gcmp2 = train_model(X2_tr, y2_tr, X2_te, y2_te,
                    f"Global — WITH Sentiment [{horizon_key}]",
                    f"ml/models/model_with_sentiment{h_suffix}.pkl",
                    saved_params.get(gws_key) or saved_params.get("global_with_sent"),
                    use_balance=horizon_cfg["balance"],
                    n_cv_splits=3)
        best_params_log[gws_key] = gp2
        if gcmp2:
            lgbm_comparisons[gws_key] = gcmp2

        # ── Results table for this horizon ─────
        if all_metrics_no_sent and all_metrics_with_sent:
            print(f"\n{'='*65}")
            print(f"RESULTS — {horizon_key}")
            print(f"{'Ticker':<7} {'No-Sent Acc':>12} {'No-Sent F1':>11} "
                  f"{'W-Sent Acc':>11} {'W-Sent F1':>10}")
            print("-" * 55)
            for m1, m2 in zip(all_metrics_no_sent, all_metrics_with_sent):
                print(f"{m1['ticker']:<7} "
                      f"{m1['accuracy']*100:>11.1f}% "
                      f"{m1['f1']*100:>10.1f}% "
                      f"{m2['accuracy']*100:>10.1f}% "
                      f"{m2['f1']*100:>9.1f}%")

            def avg(lst, key): return sum(d[key] for d in lst) / len(lst)
            m1_avg = {k: avg(all_metrics_no_sent,  k) for k in ["accuracy","f1"]}
            m2_avg = {k: avg(all_metrics_with_sent, k) for k in ["accuracy","f1"]}
            print(f"{'AVG':<7} "
                  f"{m1_avg['accuracy']*100:>11.1f}% "
                  f"{m1_avg['f1']*100:>10.1f}% "
                  f"{m2_avg['accuracy']*100:>10.1f}% "
                  f"{m2_avg['f1']*100:>9.1f}%")
            print("=" * 65)

    # ── Save artefacts ─────────────────────────
    feature_info = {
        "features_no_sentiment":   FEATURES_NO_SENT,
        "features_with_sentiment": FEATURES_WITH_SENT,
        "horizons":                list(HORIZONS.keys()),
        "trained_at":              datetime.now().isoformat(),
    }
    with open("ml/models/feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)
    print("\nSaved ml/models/feature_info.json")

    with open("ml/models/best_params.json", "w") as f:
        json.dump(best_params_log, f, indent=2)
    print("Saved ml/models/best_params.json")

    # ── LightGBM comparison summary ────────────
    if lgbm_comparisons:
        lgb_wins = sum(1 for c in lgbm_comparisons.values() if c["winner"] == "LGB")
        xgb_wins = len(lgbm_comparisons) - lgb_wins

        print(f"\n{'='*65}")
        print(f"  XGB vs LGB SUMMARY  ({lgb_wins} LGB wins / {xgb_wins} XGB wins / {len(lgbm_comparisons)} total)")
        print(f"  {'Model Key':<35} {'XGB Acc':>8} {'LGB Acc':>8} {'Winner':>8}")
        print(f"  {'-'*61}")
        for key, c in lgbm_comparisons.items():
            marker = " <--" if c["winner"] == "LGB" else ""
            print(f"  {key:<35} {c['xgb_acc']*100:>7.1f}% {c['lgb_acc']*100:>7.1f}%"
                  f"  {c['winner']:>5}{marker}")
        print(f"{'='*65}")
        print(f"\n  XGBoost models saved to standard paths (active).")
        print(f"  LightGBM models staged in *_lgbm.pkl files (inactive).")
        print(f"  To promote LGB winners: python ml/train_model.py --apply-lgbm")

        with open("ml/models/lgbm_comparison.json", "w") as f:
            json.dump(lgbm_comparisons, f, indent=2)
        print(f"  Full comparison saved -> ml/models/lgbm_comparison.json")


# ─────────────────────────────────────────────
# Apply LightGBM winners
# ─────────────────────────────────────────────

def apply_lgbm_winners():
    """
    Read lgbm_comparison.json, copy *_lgbm.pkl -> standard path
    for every model where LightGBM won.  XGBoost files are backed
    up with a _xgb.pkl suffix before being replaced.
    """
    import shutil
    comp_path = "ml/models/lgbm_comparison.json"
    if not os.path.exists(comp_path):
        print("[ERROR] lgbm_comparison.json not found — run training first.")
        return

    with open(comp_path) as f:
        comparisons = json.load(f)

    lgb_wins = {k: v for k, v in comparisons.items() if v["winner"] == "LGB"}
    if not lgb_wins:
        print("No LightGBM wins found — nothing to apply.")
        return

    print(f"\nApplying {len(lgb_wins)} LightGBM winner(s)...")
    for key, c in lgb_wins.items():
        lgbm_path = c["lgbm_path"]
        xgb_path  = c["xgb_path"]

        if not os.path.exists(lgbm_path):
            print(f"  [SKIP] {lgbm_path} not found")
            continue

        backup = xgb_path.replace(".pkl", "_xgb.pkl")
        shutil.copy2(xgb_path, backup)
        shutil.copy2(lgbm_path, xgb_path)
        print(f"  {key}: XGB backed up -> {backup}")
        print(f"  {key}: LGB promoted  -> {xgb_path}")

    print("\nDone. To revert: rename *_xgb.pkl files back to *.pkl")


def show_model_results():
    """
    Evaluate every active model on its test set and print a results table.
    Shows model type (XGB / LGB), accuracy, precision, recall, F1.
    """
    conn = get_connection()
    df   = load_features(conn)
    conn.close()

    sent_cols = ["sentiment_today", "sentiment_lag1", "sentiment_lag2",
                 "sentiment_ma_7", "sentiment_std_7", "sentiment_momentum"]
    tickers   = sorted(df["ticker"].unique())

    for horizon_key, horizon_cfg in HORIZONS.items():
        days     = horizon_cfg["days"]
        cutoff   = horizon_cfg["cutoff"]
        h_suffix = "" if horizon_key == "1d" else f"_{horizon_key}"

        df_h, _ = compute_target(df, horizon_key, horizon_cfg, tickers, days, cutoff)

        df_h = df_h.dropna(subset=["target"])
        df_h["target"] = df_h["target"].astype(int)

        rows_ns, rows_ws = [], []

        for ticker in tickers:
            t_df = df_h[df_h["ticker"] == ticker].copy()

            for variant, features, sfx, row_list in [
                ("no_sentiment",   FEATURES_NO_SENT,   "no_sentiment",   rows_ns),
                ("with_sentiment", FEATURES_WITH_SENT, "with_sentiment", rows_ws),
            ]:
                model_path = f"ml/models/{ticker}_{sfx}{h_suffix}.pkl"
                if not os.path.exists(model_path):
                    continue

                if sfx == "with_sentiment":
                    ev_df = t_df.copy()
                    ev_df[sent_cols] = ev_df[sent_cols].fillna(0.0)
                else:
                    ev_df = t_df.copy()

                ev_df = ev_df[features + ["date", "target"]].dropna()
                _, X_te, _, y_te = time_split(ev_df, features, cutoff)
                if len(X_te) == 0:
                    continue

                with open(model_path, "rb") as f:
                    mdl = pickle.load(f)

                mtype  = "LGB" if "LGBM" in type(mdl).__name__ else "XGB"
                y_pred = mdl.predict(X_te)
                row_list.append({
                    "ticker": ticker,
                    "type":   mtype,
                    "acc":    accuracy_score(y_te, y_pred),
                    "prec":   precision_score(y_te, y_pred, zero_division=0),
                    "rec":    recall_score(y_te, y_pred, zero_division=0),
                    "f1":     f1_score(y_te, y_pred, zero_division=0),
                })

        print(f"\n{'='*78}")
        print(f"  HORIZON: {horizon_key}")
        print(f"{'='*78}")
        print(f"  {'Ticker':<7} {'Type':>4}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}"
              f"    {'Type':>4}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
        print(f"  {'':7} {'--- No Sentiment ---':^37}    {'--- With Sentiment ---':^37}")
        print(f"  {'-'*76}")

        for rn, rw in zip(rows_ns, rows_ws):
            print(f"  {rn['ticker']:<7} {rn['type']:>4}  "
                  f"{rn['acc']*100:>6.1f}%  {rn['prec']*100:>6.1f}%  "
                  f"{rn['rec']*100:>6.1f}%  {rn['f1']*100:>6.1f}%"
                  f"    {rw['type']:>4}  "
                  f"{rw['acc']*100:>6.1f}%  {rw['prec']*100:>6.1f}%  "
                  f"{rw['rec']*100:>6.1f}%  {rw['f1']*100:>6.1f}%")

        if rows_ns and rows_ws:
            def avg(lst, k): return sum(r[k] for r in lst) / len(lst)
            print(f"  {'-'*76}")
            print(f"  {'AVG':<7} {'':>4}  "
                  f"{avg(rows_ns,'acc')*100:>6.1f}%  {avg(rows_ns,'prec')*100:>6.1f}%  "
                  f"{avg(rows_ns,'rec')*100:>6.1f}%  {avg(rows_ns,'f1')*100:>6.1f}%"
                  f"    {'':>4}  "
                  f"{avg(rows_ws,'acc')*100:>6.1f}%  {avg(rows_ws,'prec')*100:>6.1f}%  "
                  f"{avg(rows_ws,'rec')*100:>6.1f}%  {avg(rows_ws,'f1')*100:>6.1f}%")
        print(f"{'='*78}")


def show_accuracy_summary():
    """
    Compact accuracy-only table across all horizons.
    Columns: Ticker | 1d NS | 1d WS | 5d NS | 5d WS | 21d NS | 21d WS | 252d NS | 252d WS
    Each cell shows  ACC% (XGB/LGB)
    """
    conn = get_connection()
    df   = load_features(conn)
    conn.close()

    sent_cols = ["sentiment_today", "sentiment_lag1", "sentiment_lag2",
                 "sentiment_ma_7", "sentiment_std_7", "sentiment_momentum"]
    tickers   = sorted(df["ticker"].unique())

    # Collect results[ticker][horizon_variant] = (acc, model_type)
    results = {t: {} for t in tickers}

    for horizon_key, horizon_cfg in HORIZONS.items():
        days     = horizon_cfg["days"]
        cutoff   = horizon_cfg["cutoff"]
        h_suffix = "" if horizon_key == "1d" else f"_{horizon_key}"

        df_h, _ = compute_target(df, horizon_key, horizon_cfg, tickers, days, cutoff)

        df_h = df_h.dropna(subset=["target"])
        df_h["target"] = df_h["target"].astype(int)

        for ticker in tickers:
            t_df = df_h[df_h["ticker"] == ticker].copy()
            for label, features, sfx in [
                ("NS", FEATURES_NO_SENT,   "no_sentiment"),
                ("WS", FEATURES_WITH_SENT, "with_sentiment"),
            ]:
                model_path = f"ml/models/{ticker}_{sfx}{h_suffix}.pkl"
                if not os.path.exists(model_path):
                    results[ticker][f"{horizon_key}_{label}"] = None
                    continue
                ev_df = t_df.copy()
                if sfx == "with_sentiment":
                    ev_df[sent_cols] = ev_df[sent_cols].fillna(0.0)
                ev_df = ev_df[features + ["date", "target"]].dropna()
                _, X_te, _, y_te = time_split(ev_df, features, cutoff)
                if len(X_te) == 0:
                    results[ticker][f"{horizon_key}_{label}"] = None
                    continue
                with open(model_path, "rb") as f:
                    mdl = pickle.load(f)
                mtype = "LGB" if "LGBM" in type(mdl).__name__ else "XGB"
                acc   = accuracy_score(y_te, mdl.predict(X_te))
                results[ticker][f"{horizon_key}_{label}"] = (acc, mtype)

    # Print table
    cols = ["1d_NS", "1d_WS", "5d_NS", "5d_WS", "21d_NS", "21d_WS", "252d_NS", "252d_WS"]
    header_h = "  " + f"{'Ticker':<7}"
    for c in cols:
        header_h += f"  {c:>12}"
    print("\n" + "=" * (len(header_h) + 2))
    print("  Accuracy Summary  (NS = No Sentiment, WS = With Sentiment)")
    print("=" * (len(header_h) + 2))
    print(header_h)
    print("  " + "-" * (len(header_h) - 2))

    for ticker in tickers:
        row = f"  {ticker:<7}"
        for c in cols:
            val = results[ticker].get(c)
            if val is None:
                row += f"  {'—':>12}"
            else:
                acc, mtype = val
                cell = f"{acc*100:.1f}% ({mtype})"
                row += f"  {cell:>12}"
        print(row)

    # Averages
    print("  " + "-" * (len(header_h) - 2))
    avg_row = f"  {'AVG':<7}"
    for c in cols:
        vals = [results[t][c][0] for t in tickers if results[t].get(c)]
        avg_row += f"  {(sum(vals)/len(vals)*100):>10.1f}%" if vals else f"  {'—':>12}"
    print(avg_row)
    print("=" * (len(header_h) + 2))


if __name__ == "__main__":
    if "--apply-lgbm" in sys.argv:
        apply_lgbm_winners()
    elif "--summary" in sys.argv:
        show_accuracy_summary()
    elif "--results" in sys.argv:
        show_model_results()
    else:
        ticker_arg  = None
        horizon_arg = None
        if "--ticker" in sys.argv:
            idx = sys.argv.index("--ticker")
            if idx + 1 < len(sys.argv):
                ticker_arg = sys.argv[idx + 1]
        if "--horizon" in sys.argv:
            idx = sys.argv.index("--horizon")
            if idx + 1 < len(sys.argv):
                horizon_arg = sys.argv[idx + 1]
        run_training(ticker_filter=ticker_arg, horizon_filter=horizon_arg)