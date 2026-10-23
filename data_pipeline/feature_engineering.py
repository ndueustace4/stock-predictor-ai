# data_pipeline/feature_engineering.py
# ─────────────────────────────────────────────
# Phase 3 — Feature Engineering
# Builds the ML-ready dataset from stock prices
# and sentiment scores, saves to features table.
#
# New features added (v2):
#   Technical : RSI-14, MACD, MACD signal, Bollinger Band width,
#               volume change %, price/MA20 ratio
#   Sentiment : sentiment_lag2, sentiment_ma_7,
#               sentiment_std_7, sentiment_momentum
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
import sys
import os
import yfinance as yf
from psycopg2.extras import execute_values

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import STOCKS
from data_pipeline.db_connection import get_connection


def _yf_close(symbol: str, start: str, auto_adjust: bool = True) -> pd.Series:
    """Download Close prices for one symbol, handling yfinance 1.2+ MultiIndex columns."""
    try:
        df = yf.download(symbol, start=start, progress=False,
                         auto_adjust=auto_adjust)
        if df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        return df["Close"].squeeze()
    except Exception:
        return pd.Series(dtype=float)

# ── Market context cache (downloaded once per session) ───────────
_market_cache: dict = {}

def load_market_features() -> pd.DataFrame:
    """
    Download SPY, VIX, and QQQ — merge into one DataFrame indexed by date.
    Cached for the session so each ticker doesn't re-download.

    Columns returned:
      spy_return_1d  — S&P 500 daily return  (broad market)
      qqq_return_1d  — Nasdaq-100 daily return (tech sector context)
      vix_level      — CBOE VIX closing level  (market fear)
      vix_change     — day-over-day VIX change  (fear rising/falling)
    """
    if "data" not in _market_cache:
        print("  Downloading market context data (SPY, QQQ, VIX, HSI, MCHI, sector ETFs)...")
        start = "2014-01-01"

        spy = _yf_close("SPY",   start, auto_adjust=True)
        qqq = _yf_close("QQQ",   start, auto_adjust=True)
        vix = _yf_close("^VIX",  start, auto_adjust=False)
        if vix.empty:
            print("  [WARN] VIX download failed — filling with zeros")

        spy_ret = spy.pct_change().round(6)
        qqq_ret = qqq.pct_change().round(6)
        vix_lvl = vix.round(4)        if not vix.empty else pd.Series(0.0, index=spy.index)
        vix_chg = vix.diff().round(4) if not vix.empty else pd.Series(0.0, index=spy.index)

        sector_rets = {}
        for sym in ["XLK", "XLF", "XLE", "XLV", "XLI"]:
            col = f"{sym.lower()}_return_1d"
            etf = _yf_close(sym, start, auto_adjust=True)
            if etf.empty:
                print(f"  [WARN] {sym} download failed — filling with zeros")
                sector_rets[col] = pd.Series(0.0, index=spy.index)
            else:
                sector_rets[col] = etf.pct_change().round(6)

        # Chinese market context (primarily for BABA)
        hsi = _yf_close("^HSI",  start, auto_adjust=False)
        if hsi.empty:
            print("  [WARN] HSI download failed — filling with zeros")
        hsi_ret = hsi.pct_change().round(6) if not hsi.empty else pd.Series(dtype=float)

        mchi = _yf_close("MCHI", start, auto_adjust=True)
        if mchi.empty:
            print("  [WARN] MCHI download failed — filling with zeros")
        mchi_ret = mchi.pct_change().round(6) if not mchi.empty else pd.Series(dtype=float)

        # CNY/USD exchange rate — yuan strength signal (key driver for BABA)
        cny = _yf_close("CNY=X", start, auto_adjust=False)
        if cny.empty:
            print("  [WARN] CNY/USD download failed — filling with zeros")
        cny_ret = cny.pct_change().round(6) if not cny.empty else pd.Series(dtype=float)

        mkt = pd.DataFrame({
            "spy_return_1d":  spy_ret,
            "qqq_return_1d":  qqq_ret,
            "vix_level":      vix_lvl,
            "vix_change":     vix_chg,
            "hsi_return_1d":  hsi_ret  if not hsi_ret.empty  else pd.Series(0.0, index=spy.index),
            "mchi_return_1d": mchi_ret if not mchi_ret.empty else pd.Series(0.0, index=spy.index),
            "cny_usd_return": cny_ret  if not cny_ret.empty  else pd.Series(0.0, index=spy.index),
            **sector_rets,
        })
        mkt.index = pd.to_datetime(mkt.index).normalize()
        mkt = mkt.reset_index().rename(columns={"index": "date", "Date": "date"})
        _market_cache["data"] = mkt
        print(f"  [OK] Market data loaded ({len(mkt)} days)")
    return _market_cache["data"]


# ─────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────

def load_stock_prices(ticker: str, conn) -> pd.DataFrame:
    sql = """
        SELECT date, open, high, low, close, volume
        FROM stock_prices
        WHERE ticker = %s
        ORDER BY date ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker,))
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"]   = pd.to_datetime(df["date"])
    df["open"]   = df["open"].astype(float)
    df["high"]   = df["high"].astype(float)
    df["low"]    = df["low"].astype(float)
    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


def load_sentiment(ticker: str, conn) -> pd.DataFrame:
    """
    Load daily average sentiment for a ticker.
    Prefers FinBERT score (groq_sentiment); falls back to VADER (sentiment_score).
    """
    sql = """
        SELECT date,
               AVG(COALESCE(groq_sentiment, sentiment_score)) AS avg_sentiment
        FROM news_sentiment
        WHERE ticker = %s
          AND COALESCE(groq_sentiment, sentiment_score) IS NOT NULL
        GROUP BY date
        ORDER BY date ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker,))
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["date", "avg_sentiment"])
    df["date"]          = pd.to_datetime(df["date"])
    df["avg_sentiment"] = df["avg_sentiment"].astype(float)
    return df


def load_earnings_surprise(ticker: str) -> pd.DataFrame:
    """
    Fetch quarterly earnings surprise % from Yahoo Finance.
    Returns DataFrame with date + earnings_surprise columns.
    In build_features() this is forward-filled so every trading day
    carries the most recent past earnings surprise value.
    Only covers ~2-3 years of history (Yahoo limit) — older rows get 0.0.
    """
    try:
        ed = yf.Ticker(ticker).earnings_dates
        if ed is None or ed.empty:
            return pd.DataFrame()

        ed = ed.reset_index()
        surp_col = next((c for c in ed.columns if "Surprise" in str(c)), None)
        date_col = ed.columns[0]
        if surp_col is None:
            return pd.DataFrame()

        ed = ed[[date_col, surp_col]].rename(
            columns={date_col: "date", surp_col: "earnings_surprise"}
        )
        ed["date"] = pd.to_datetime(ed["date"]).dt.tz_localize(None).dt.normalize()
        ed = ed.dropna(subset=["earnings_surprise"])
        ed = ed[ed["date"] <= pd.Timestamp.today()]
        ed["earnings_surprise"] = (ed["earnings_surprise"] / 100).round(4)
        return ed.sort_values("date").reset_index(drop=True)

    except Exception as e:
        print(f"  [WARN] Earnings data unavailable for {ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# Fundamental & macro loaders
# ─────────────────────────────────────────────

_macro_cache: dict = {}

def load_fundamentals(ticker: str, conn) -> pd.DataFrame:
    """Load daily-forward-filled fundamentals for one ticker."""
    sql = """
        SELECT date, eps_ttm, revenue_ttm, revenue_growth_yoy, eps_growth_yoy
        FROM fundamentals
        WHERE ticker = %s
        ORDER BY date ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker,))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "eps_ttm", "revenue_ttm",
                                     "revenue_growth_yoy", "eps_growth_yoy"])
    df["date"] = pd.to_datetime(df["date"])
    for col in ["eps_ttm", "revenue_ttm", "revenue_growth_yoy", "eps_growth_yoy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_macro_rates(conn) -> pd.DataFrame:
    """Load Treasury yield data — cached for the session."""
    if "data" not in _macro_cache:
        sql = "SELECT date, treasury_10y, treasury_2y, yield_curve FROM macro_rates ORDER BY date ASC"
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["date", "treasury_10y", "treasury_2y", "yield_curve"])
            df["date"] = pd.to_datetime(df["date"])
            for col in ["treasury_10y", "treasury_2y", "yield_curve"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            _macro_cache["data"] = df
        else:
            _macro_cache["data"] = pd.DataFrame()
    return _macro_cache["data"]


# ─────────────────────────────────────────────
# Technical indicator calculators
# ─────────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).
    > 70 = overbought (price may fall), < 30 = oversold (price may rise).
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(4)


def calc_macd(series: pd.Series,
              fast: int = 12, slow: int = 26, signal: int = 9
              ) -> tuple[pd.Series, pd.Series]:
    """
    MACD = EMA(12) - EMA(26).
    Signal = EMA(9) of MACD.
    When MACD crosses above signal → bullish.
    """
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = (ema_fast - ema_slow).round(4)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean().round(4)
    return macd_line, signal_line


def calc_bollinger_width(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Bollinger Band width = (upper - lower) / middle.
    High width = high volatility. Low width = squeeze before big move.
    """
    ma      = series.rolling(window).mean()
    std     = series.rolling(window).std()
    upper   = ma + 2 * std
    lower   = ma - 2 * std
    width   = ((upper - lower) / ma.replace(0, np.nan)).round(6)
    return width


def calc_volume_change(volume: pd.Series) -> pd.Series:
    """
    Volume percentage change vs previous day.
    Unusual volume often precedes large price moves.
    """
    return volume.pct_change().round(4)


def calc_price_ma_ratio(close: pd.Series, ma: pd.Series) -> pd.Series:
    """
    How far price is from its 20-day average (as a ratio).
    > 1.0 = price above MA (momentum), < 1.0 = below MA (potential mean reversion).
    """
    return (close / ma.replace(0, np.nan)).round(6)


def calc_returns(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Percentage returns over 1, 5, and 10 days.
    Normalised moves are far more predictive than raw price differences —
    a $5 move means something very different for a $10 stock vs a $500 stock.
    """
    r1  = close.pct_change(1).round(6)
    r5  = close.pct_change(5).round(6)
    r10 = close.pct_change(10).round(6)
    return r1, r5, r10


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range normalised by close price.
    Better volatility measure than rolling std because it captures
    gap opens (overnight moves) that rolling std misses.
    Expressed as a ratio (e.g. 0.02 = ATR is 2% of price).
    """
    hl   = df["high"] - df["low"]
    hpc  = (df["high"] - df["close"].shift(1)).abs()
    lpc  = (df["low"]  - df["close"].shift(1)).abs()
    tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    atr  = tr.ewm(com=period - 1, min_periods=period).mean()
    return (atr / df["close"].replace(0, np.nan)).round(6)


def calc_high_low_ratios(df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    """
    Where is price relative to its recent high and low?

    high_20d_ratio = close / 20-day high  (1.0 = AT the high, <1 = below it)
    low_20d_ratio  = close / 20-day low   (1.0 = AT the low,  >1 = above it)

    These capture mean-reversion signals — stocks near recent highs often
    continue up (breakout) or pull back; near lows often bounce.
    """
    high_20 = df["high"].rolling(window).max()
    low_20  = df["low"].rolling(window).min()
    h_ratio = (df["close"] / high_20.replace(0, np.nan)).round(6)
    l_ratio = (df["close"] / low_20.replace(0, np.nan)).round(6)
    return h_ratio, l_ratio


# ─────────────────────────────────────────────
# Sentiment feature builder
# ─────────────────────────────────────────────

def build_sentiment_features(sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Expand a daily sentiment series into multiple lag/rolling features.

    Returns a DataFrame with date as index and columns:
      sentiment_lag1    — yesterday's score
      sentiment_lag2    — 2 days ago
      sentiment_ma_7    — 7-day rolling mean (smoothed signal)
      sentiment_std_7   — 7-day rolling std  (uncertainty/disagreement)
      sentiment_momentum — lag1 minus lag2   (is sentiment improving?)
    """
    s = sentiment.set_index("date")["avg_sentiment"].sort_index()

    out = pd.DataFrame(index=s.index)
    out["sentiment_raw"]      = s
    out["sentiment_today"]    = s.shift(-1).round(4)  # same-day news (shift(-1) cancels the +1day shift below)
    out["sentiment_lag1"]     = s.shift(1).round(4)
    out["sentiment_lag2"]     = s.shift(2).round(4)
    out["sentiment_ma_7"]     = s.rolling(7).mean().round(4)
    out["sentiment_std_7"]    = s.rolling(7).std().round(4)
    out["sentiment_momentum"] = (out["sentiment_lag1"] - out["sentiment_lag2"]).round(4)

    out = out.drop(columns=["sentiment_raw"]).reset_index()
    # Shift everything by 1 so features for date D use news up to D-1
    # (sentiment_today uses shift(-1) above to cancel this, landing on same-day news)
    out["date"] = out["date"] + pd.Timedelta(days=1)
    return out


# ─────────────────────────────────────────────
# Main feature builder
# ─────────────────────────────────────────────

def build_features(ticker: str, conn) -> pd.DataFrame:
    """
    Build the full feature set for one ticker.

    Features:
    ─────────────────────────────────────
    Price lags   : price_lag1, price_lag2
    Moving avgs  : ma_5, ma_10, ma_20
    Volatility   : volatility_7, bollinger_width
    Momentum     : momentum_5, macd, macd_signal
    Oscillator   : rsi_14
    Volume       : volume_change
    Ratio        : price_ma_ratio
    Sentiment    : sentiment_lag1, sentiment_lag2,
                   sentiment_ma_7, sentiment_std_7,
                   sentiment_momentum
    Target       : 1 = price up tomorrow, 0 = down
    ─────────────────────────────────────
    """
    print(f"  Building features for {ticker}...")

    prices    = load_stock_prices(ticker, conn)
    sentiment = load_sentiment(ticker, conn)

    if prices.empty:
        print(f"  [WARNING] No price data for {ticker}")
        return pd.DataFrame()

    df = prices.copy()

    # ── Original features ──────────────────────
    df["price_lag1"]  = df["close"].shift(1)
    df["price_lag2"]  = df["close"].shift(2)
    df["price_lag3"]  = df["close"].shift(3)
    df["price_lag5"]  = df["close"].shift(5)
    df["ma_5"]        = df["close"].rolling(5).mean()
    df["ma_10"]       = df["close"].rolling(10).mean()
    df["ma_20"]       = df["close"].rolling(20).mean()
    df["volatility_7"]= df["close"].rolling(7).std()
    df["momentum_5"]  = df["close"] - df["close"].shift(5)

    # ── New technical indicators ───────────────
    df["rsi_14"]          = calc_rsi(df["close"], 14)
    df["macd"], \
    df["macd_signal"]     = calc_macd(df["close"])
    df["bollinger_width"] = calc_bollinger_width(df["close"])
    df["volume_change"]   = calc_volume_change(df["volume"])
    df["price_ma_ratio"]  = calc_price_ma_ratio(df["close"], df["ma_20"])

    # ── v3 features ────────────────────────────
    df["return_1d"], \
    df["return_5d"], \
    df["return_10d"]      = calc_returns(df["close"])
    df["day_of_week"]     = df["date"].dt.dayofweek            # 0=Mon 4=Fri
    df["atr_14"]          = calc_atr(df, period=14)
    df["high_20d_ratio"], \
    df["low_20d_ratio"]   = calc_high_low_ratios(df, window=20)

    # ── Market context features ───────────────
    mkt = load_market_features()
    df  = df.merge(mkt, on="date", how="left")
    for col in ["spy_return_1d", "qqq_return_1d", "vix_level", "vix_change",
                "hsi_return_1d", "mchi_return_1d", "cny_usd_return",
                "xlk_return_1d", "xlf_return_1d", "xle_return_1d", "xlv_return_1d", "xli_return_1d"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    # ── Long-term regime feature ──────────────
    df["ma_252"]           = df["close"].rolling(252).mean()
    df["price_ma252_ratio"] = calc_price_ma_ratio(df["close"], df["ma_252"])

    # ── Target (next-day direction) ───────────────
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # ── Sentiment features ────────────────────
    if not sentiment.empty:
        sent_features = build_sentiment_features(sentiment)
        df = df.merge(sent_features, on="date", how="left")
    else:
        for col in ["sentiment_today", "sentiment_lag1", "sentiment_lag2",
                    "sentiment_ma_7", "sentiment_std_7", "sentiment_momentum"]:
            df[col] = np.nan

    # ── Earnings surprise ─────────────────────
    earnings = load_earnings_surprise(ticker)
    if not earnings.empty:
        df = df.merge(earnings, on="date", how="left")
        df["earnings_surprise"] = df["earnings_surprise"].ffill().fillna(0.0)
    else:
        df["earnings_surprise"] = 0.0

    # ── Fundamental features ──────────────────
    fundamentals = load_fundamentals(ticker, conn)
    if not fundamentals.empty:
        df = df.merge(fundamentals, on="date", how="left")
        df["pe_ratio"] = (df["close"] / df["eps_ttm"].replace(0, np.nan)).round(2)
        for col in ["pe_ratio", "revenue_growth_yoy", "eps_growth_yoy"]:
            df[col] = df[col].ffill().fillna(0.0)
        # Clamp extremes: pe_ratio to [-500, 500], growth rates to [-10, 10]
        df["pe_ratio"]            = df["pe_ratio"].clip(-500, 500)
        df["revenue_growth_yoy"]  = df["revenue_growth_yoy"].clip(-10, 10)
        df["eps_growth_yoy"]      = df["eps_growth_yoy"].clip(-10, 10)
    else:
        df["pe_ratio"]            = 0.0
        df["revenue_growth_yoy"]  = 0.0
        df["eps_growth_yoy"]      = 0.0

    # ── Macro / Treasury yields ───────────────
    macro = load_macro_rates(conn)
    if not macro.empty:
        df = df.merge(macro, on="date", how="left")
        for col in ["treasury_10y", "treasury_2y", "yield_curve"]:
            df[col] = df[col].ffill().fillna(0.0)
    else:
        df["treasury_10y"] = 0.0
        df["treasury_2y"]  = 0.0
        df["yield_curve"]  = 0.0

    df["ticker"] = ticker

    # ── Drop rows missing critical features ───
    df = df.dropna(subset=["ma_20", "price_lag2", "rsi_14", "macd", "target"])

    # ── Round numerics ─────────────────────────
    numeric_cols = [
        "close", "ma_5", "ma_10", "ma_20",
        "volatility_7", "momentum_5", "price_lag1", "price_lag2",
        "rsi_14", "macd", "macd_signal", "bollinger_width",
        "volume_change", "price_ma_ratio",
    ]
    df[numeric_cols] = df[numeric_cols].round(4)

    print(f"  [OK] {len(df)} feature rows built for {ticker}")
    return df


# ─────────────────────────────────────────────
# DB save
# ─────────────────────────────────────────────

def save_features_to_db(df: pd.DataFrame, ticker: str, conn) -> int:
    if df.empty:
        return 0

    rows = []
    for _, row in df.iterrows():
        rows.append((
            row["ticker"],
            row["date"].date(),
            row["close"],
            int(row["volume"]),
            row["ma_5"],
            row["ma_10"],
            row["ma_20"],
            row["volatility_7"],
            row["momentum_5"],
            row["price_lag1"],
            row["price_lag2"],
            # technical v2
            row.get("rsi_14"),
            row.get("macd"),
            row.get("macd_signal"),
            row.get("bollinger_width"),
            row.get("volume_change"),
            row.get("price_ma_ratio"),
            # sentiment
            row.get("sentiment_today"),
            row.get("sentiment_lag1"),
            row.get("sentiment_lag2"),
            row.get("sentiment_ma_7"),
            row.get("sentiment_std_7"),
            row.get("sentiment_momentum"),
            # v3 new features
            row.get("return_1d"),
            row.get("return_5d"),
            row.get("return_10d"),
            int(row["day_of_week"]) if pd.notna(row.get("day_of_week")) else None,
            row.get("atr_14"),
            row.get("high_20d_ratio"),
            row.get("low_20d_ratio"),
            row.get("price_lag3"),
            row.get("price_lag5"),
            row.get("spy_return_1d"),
            row.get("qqq_return_1d"),
            row.get("vix_level"),
            row.get("vix_change"),
            row.get("hsi_return_1d"),
            row.get("mchi_return_1d"),
            row.get("cny_usd_return"),
            row.get("xlk_return_1d"),
            row.get("xlf_return_1d"),
            row.get("xle_return_1d"),
            row.get("xlv_return_1d"),
            row.get("xli_return_1d"),
            row.get("price_ma252_ratio"),
            row.get("earnings_surprise"),
            row.get("pe_ratio"),
            row.get("revenue_growth_yoy"),
            row.get("eps_growth_yoy"),
            row.get("treasury_10y"),
            row.get("treasury_2y"),
            row.get("yield_curve"),
            int(row["target"]),
        ))

    sql = """
        INSERT INTO features (
            ticker, date, close, volume,
            ma_5, ma_10, ma_20,
            volatility_7, momentum_5,
            price_lag1, price_lag2,
            rsi_14, macd, macd_signal,
            bollinger_width, volume_change, price_ma_ratio,
            sentiment_today,
            sentiment_lag1, sentiment_lag2,
            sentiment_ma_7, sentiment_std_7, sentiment_momentum,
            return_1d, return_5d, return_10d,
            day_of_week, atr_14,
            high_20d_ratio, low_20d_ratio,
            price_lag3, price_lag5,
            spy_return_1d, qqq_return_1d,
            vix_level, vix_change,
            hsi_return_1d, mchi_return_1d, cny_usd_return,
            xlk_return_1d, xlf_return_1d, xle_return_1d, xlv_return_1d, xli_return_1d,
            price_ma252_ratio,
            earnings_surprise,
            pe_ratio, revenue_growth_yoy, eps_growth_yoy,
            treasury_10y, treasury_2y, yield_curve,
            target
        )
        VALUES %s
        ON CONFLICT (ticker, date) DO UPDATE SET
            close              = EXCLUDED.close,
            volume             = EXCLUDED.volume,
            ma_5               = EXCLUDED.ma_5,
            ma_10              = EXCLUDED.ma_10,
            ma_20              = EXCLUDED.ma_20,
            volatility_7       = EXCLUDED.volatility_7,
            momentum_5         = EXCLUDED.momentum_5,
            price_lag1         = EXCLUDED.price_lag1,
            price_lag2         = EXCLUDED.price_lag2,
            rsi_14             = EXCLUDED.rsi_14,
            macd               = EXCLUDED.macd,
            macd_signal        = EXCLUDED.macd_signal,
            bollinger_width    = EXCLUDED.bollinger_width,
            volume_change      = EXCLUDED.volume_change,
            price_ma_ratio     = EXCLUDED.price_ma_ratio,
            sentiment_today    = EXCLUDED.sentiment_today,
            sentiment_lag1     = EXCLUDED.sentiment_lag1,
            sentiment_lag2     = EXCLUDED.sentiment_lag2,
            sentiment_ma_7     = EXCLUDED.sentiment_ma_7,
            sentiment_std_7    = EXCLUDED.sentiment_std_7,
            sentiment_momentum = EXCLUDED.sentiment_momentum,
            return_1d          = EXCLUDED.return_1d,
            return_5d          = EXCLUDED.return_5d,
            return_10d         = EXCLUDED.return_10d,
            day_of_week        = EXCLUDED.day_of_week,
            atr_14             = EXCLUDED.atr_14,
            high_20d_ratio     = EXCLUDED.high_20d_ratio,
            low_20d_ratio      = EXCLUDED.low_20d_ratio,
            price_lag3         = EXCLUDED.price_lag3,
            price_lag5         = EXCLUDED.price_lag5,
            spy_return_1d      = EXCLUDED.spy_return_1d,
            qqq_return_1d      = EXCLUDED.qqq_return_1d,
            vix_level          = EXCLUDED.vix_level,
            vix_change         = EXCLUDED.vix_change,
            hsi_return_1d      = EXCLUDED.hsi_return_1d,
            mchi_return_1d     = EXCLUDED.mchi_return_1d,
            cny_usd_return     = EXCLUDED.cny_usd_return,
            xlk_return_1d      = EXCLUDED.xlk_return_1d,
            xlf_return_1d      = EXCLUDED.xlf_return_1d,
            xle_return_1d      = EXCLUDED.xle_return_1d,
            xlv_return_1d      = EXCLUDED.xlv_return_1d,
            xli_return_1d      = EXCLUDED.xli_return_1d,
            price_ma252_ratio  = EXCLUDED.price_ma252_ratio,
            earnings_surprise  = EXCLUDED.earnings_surprise,
            pe_ratio           = EXCLUDED.pe_ratio,
            revenue_growth_yoy = EXCLUDED.revenue_growth_yoy,
            eps_growth_yoy     = EXCLUDED.eps_growth_yoy,
            treasury_10y       = EXCLUDED.treasury_10y,
            treasury_2y        = EXCLUDED.treasury_2y,
            yield_curve        = EXCLUDED.yield_curve,
            target             = EXCLUDED.target
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    return len(rows)


# ─────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────

def run_feature_engineering():
    print("=" * 55)
    print("Phase 3 — Feature Engineering (v2)")
    print(f"Stocks : {STOCKS}")
    print("New    : RSI-14, MACD, Bollinger width, volume change,")
    print("         price/MA ratio, sentiment lag2/ma7/std7/momentum")
    print("=" * 55)

    conn        = get_connection()

    # Ensure new columns exist (safe to run repeatedly)
    with conn.cursor() as cur:
        for col_def in [
            "spy_return_1d NUMERIC(10, 6)",
            "qqq_return_1d NUMERIC(10, 6)",
            "vix_level     NUMERIC(8,  4)",
            "vix_change    NUMERIC(8,  4)",
            "hsi_return_1d     NUMERIC(10, 6)",
            "mchi_return_1d    NUMERIC(10, 6)",
            "cny_usd_return    NUMERIC(10, 6)",
            "xlk_return_1d     NUMERIC(10, 6)",
            "xlf_return_1d     NUMERIC(10, 6)",
            "xle_return_1d     NUMERIC(10, 6)",
            "xlv_return_1d     NUMERIC(10, 6)",
            "xli_return_1d     NUMERIC(10, 6)",
            "price_ma252_ratio NUMERIC(10, 6)",
            "earnings_surprise NUMERIC(8, 4)",
            "sentiment_today   NUMERIC(8, 4)",
            "pe_ratio           FLOAT",
            "revenue_growth_yoy FLOAT",
            "eps_growth_yoy     FLOAT",
            "treasury_10y       FLOAT",
            "treasury_2y        FLOAT",
            "yield_curve        FLOAT",
        ]:
            cur.execute(f"ALTER TABLE features ADD COLUMN IF NOT EXISTS {col_def}")
        # Convert any previously-created NUMERIC versions to FLOAT
        for col in ["pe_ratio", "revenue_growth_yoy", "eps_growth_yoy",
                    "treasury_10y", "treasury_2y", "yield_curve"]:
            cur.execute(f"""
                DO $$ BEGIN
                    ALTER TABLE features ALTER COLUMN {col} TYPE FLOAT;
                EXCEPTION WHEN others THEN NULL;
                END $$;
            """)
    conn.commit()

    total_saved = 0

    for ticker in STOCKS:
        df    = build_features(ticker, conn)
        saved = save_features_to_db(df, ticker, conn)
        total_saved += saved
        print(f"  Saved {saved} feature rows for {ticker}\n")

    conn.close()
    print("=" * 55)
    print(f"Done! Total feature rows saved: {total_saved}")
    print("=" * 55)


def show_feature_sample():
    conn = get_connection()
    sql  = """
        SELECT ticker, date, close, rsi_14, macd,
               bollinger_width, sentiment_lag1,
               sentiment_ma_7, target
        FROM features
        WHERE rsi_14 IS NOT NULL
        ORDER BY ticker, date DESC
        LIMIT 5
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    conn.close()

    print("\nSample feature rows (new indicators):")
    print(f"{'Ticker':<8} {'Date':<12} {'Close':>8} {'RSI':>7} "
          f"{'MACD':>8} {'BollW':>8} {'Sent1':>7} {'SentMA7':>8} {'Target':>7}")
    print("-" * 80)
    for row in rows:
        ticker, date, close, rsi, macd, boll, s1, sma7, target = row
        def f(v, fmt=".2f"): return format(float(v), fmt) if v is not None else "  None"
        print(f"{ticker:<8} {str(date):<12} {f(close):>8} {f(rsi):>7} "
              f"{f(macd):>8} {f(boll, '.4f'):>8} {f(s1):>7} {f(sma7):>8} {target:>7}")


if __name__ == "__main__":
    run_feature_engineering()
    show_feature_sample()