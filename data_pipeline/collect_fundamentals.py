# data_pipeline/collect_fundamentals.py
# ─────────────────────────────────────────────
# Fetches quarterly EPS and revenue from Alpha Vantage
# (goes back to 2009 — far more history than yfinance).
#
# Endpoints used:
#   INCOME_STATEMENT  — quarterly revenue
#   EARNINGS          — quarterly EPS (uses reportedDate to avoid look-ahead)
#
# Rate limits (free tier): 25 req/day, 5 req/min
# 10 tickers × 2 calls = 20 calls — fits in one run with 13s sleep between calls.
# ─────────────────────────────────────────────

import sys
import os
import time
import requests
import pandas as pd
from psycopg2.extras import execute_values

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import STOCKS, ALPHA_KEY
from data_pipeline.db_connection import get_connection

AV_BASE          = "https://www.alphavantage.co/query"
RATE_LIMIT_SLEEP = 13   # seconds between calls (keeps us under 5 req/min)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker             VARCHAR(10) NOT NULL,
    date               DATE        NOT NULL,
    eps_ttm            FLOAT,
    revenue_ttm        FLOAT,
    revenue_growth_yoy FLOAT,
    eps_growth_yoy     FLOAT,
    PRIMARY KEY (ticker, date)
);
"""


# ─────────────────────────────────────────────
# Alpha Vantage fetchers
# ─────────────────────────────────────────────

def _av_get(function: str, symbol: str) -> dict:
    url = f"{AV_BASE}?function={function}&symbol={symbol}&apikey={ALPHA_KEY}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_revenue(ticker: str) -> pd.DataFrame:
    """
    Quarterly revenue from INCOME_STATEMENT.
    Returns DataFrame with columns: date, revenue  (oldest first).
    Uses fiscalDateEnding as the date.
    """
    data = _av_get("INCOME_STATEMENT", ticker)
    reports = data.get("quarterlyReports", [])
    if not reports:
        print(f"  [WARN] No income statement data for {ticker}")
        return pd.DataFrame()

    records = []
    for r in reports:
        try:
            rev = r.get("totalRevenue")
            if rev in (None, "None", ""):
                continue
            records.append({
                "date":    pd.to_datetime(r["fiscalDateEnding"]),
                "revenue": float(rev),
            })
        except (ValueError, TypeError, KeyError):
            continue

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"  INCOME_STATEMENT: {len(df)} quarters  "
          f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")
    return df


def fetch_eps(ticker: str) -> pd.DataFrame:
    """
    Quarterly EPS from EARNINGS.
    Uses reportedDate (when earnings were announced) to avoid look-ahead bias.
    Returns DataFrame with columns: date, eps  (oldest first).
    """
    data = _av_get("EARNINGS", ticker)
    reports = data.get("quarterlyEarnings", [])
    if not reports:
        print(f"  [WARN] No earnings data for {ticker}")
        return pd.DataFrame()

    records = []
    for r in reports:
        try:
            eps = r.get("reportedEPS")
            if eps in (None, "None", ""):
                continue
            # Use reportedDate when available, fall back to fiscalDateEnding
            date_str = r.get("reportedDate") or r.get("fiscalDateEnding")
            if not date_str:
                continue
            records.append({
                "date": pd.to_datetime(date_str),
                "eps":  float(eps),
            })
        except (ValueError, TypeError, KeyError):
            continue

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"  EARNINGS:         {len(df)} quarters  "
          f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")
    return df


# ─────────────────────────────────────────────
# TTM + YoY computation
# ─────────────────────────────────────────────

def compute_quarterly_metrics(rev_df: pd.DataFrame, eps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge revenue + EPS by nearest quarter, compute TTM and YoY growth.
    Returns one row per quarter with: date, eps_ttm, revenue_ttm,
    revenue_growth_yoy, eps_growth_yoy.
    """
    # Align on revenue dates (use as-of merge so EPS date doesn't need exact match)
    if rev_df.empty and eps_df.empty:
        return pd.DataFrame()

    if rev_df.empty:
        base = eps_df.rename(columns={"date": "date"}).copy()
        base["revenue"] = float("nan")
    elif eps_df.empty:
        base = rev_df.copy()
        base["eps"] = float("nan")
    else:
        base = pd.merge_asof(
            rev_df.sort_values("date"),
            eps_df.sort_values("date"),
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta("45 days"),
        )

    records = []
    for i in range(len(base)):
        if i < 3:
            continue  # need 4 quarters for TTM

        w4 = base.iloc[i - 3: i + 1]
        rev_ttm = float(w4["revenue"].sum()) if "revenue" in w4 else None
        eps_ttm = float(w4["eps"].sum())     if "eps"     in w4 else None

        rev_growth = None
        eps_growth = None

        if i >= 7:
            w4_prev = base.iloc[i - 7: i - 3]
            if rev_ttm is not None:
                rev_prev = float(w4_prev["revenue"].sum())
                if rev_prev != 0 and pd.notna(rev_prev):
                    rev_growth = (rev_ttm - rev_prev) / abs(rev_prev)
            if eps_ttm is not None:
                eps_prev = float(w4_prev["eps"].sum())
                if eps_prev != 0 and pd.notna(eps_prev):
                    eps_growth = (eps_ttm - eps_prev) / abs(eps_prev)

        records.append({
            "date":               base.iloc[i]["date"],
            "eps_ttm":            eps_ttm if pd.notna(eps_ttm) else None,
            "revenue_ttm":        rev_ttm if pd.notna(rev_ttm) else None,
            "revenue_growth_yoy": rev_growth,
            "eps_growth_yoy":     eps_growth,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Forward-fill to trading days + upsert
# ─────────────────────────────────────────────

def fill_to_trading_days(ticker: str, quarterly: pd.DataFrame, conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT date FROM stock_prices WHERE ticker = %s ORDER BY date ASC",
            (ticker,)
        )
        trading_days = [r[0] for r in cur.fetchall()]

    if not trading_days:
        return pd.DataFrame()

    trading_df = pd.DataFrame({"date": pd.to_datetime(trading_days).astype("datetime64[us]")})
    quarterly["date"] = pd.to_datetime(quarterly["date"]).astype("datetime64[us]")

    merged = pd.merge_asof(
        trading_df.sort_values("date"),
        quarterly.sort_values("date"),
        on="date",
        direction="backward",
    )
    return merged


def upsert(ticker: str, df: pd.DataFrame, conn) -> int:
    if df.empty:
        return 0

    def _f(v):
        try:
            return None if pd.isna(v) else float(v)
        except (TypeError, ValueError):
            return None

    rows = [
        (ticker,
         r["date"].date() if hasattr(r["date"], "date") else r["date"],
         _f(r.get("eps_ttm")),
         _f(r.get("revenue_ttm")),
         _f(r.get("revenue_growth_yoy")),
         _f(r.get("eps_growth_yoy")))
        for _, r in df.iterrows()
    ]

    sql = """
        INSERT INTO fundamentals
            (ticker, date, eps_ttm, revenue_ttm, revenue_growth_yoy, eps_growth_yoy)
        VALUES %s
        ON CONFLICT (ticker, date) DO UPDATE SET
            eps_ttm            = EXCLUDED.eps_ttm,
            revenue_ttm        = EXCLUDED.revenue_ttm,
            revenue_growth_yoy = EXCLUDED.revenue_growth_yoy,
            eps_growth_yoy     = EXCLUDED.eps_growth_yoy
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    return len(rows)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def collect_all_fundamentals():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()

    total = 0
    for i, ticker in enumerate(STOCKS):
        print(f"\n[{ticker}]  ({i+1}/{len(STOCKS)})")

        try:
            rev_df = fetch_revenue(ticker)
            time.sleep(RATE_LIMIT_SLEEP)

            eps_df = fetch_eps(ticker)
            if i < len(STOCKS) - 1:
                time.sleep(RATE_LIMIT_SLEEP)

        except Exception as e:
            print(f"  [ERROR] API call failed: {e}")
            continue

        quarterly = compute_quarterly_metrics(rev_df, eps_df)
        if quarterly.empty:
            print("  Skipped — no quarterly data")
            continue
        print(f"  {len(quarterly)} quarterly records computed")

        daily = fill_to_trading_days(ticker, quarterly, conn)
        if daily.empty:
            print("  Skipped — no trading days found")
            continue

        n = upsert(ticker, daily, conn)
        print(f"  {n} rows upserted")
        total += n

    conn.close()
    print(f"\nTotal: {total} rows upserted across all tickers")


if __name__ == "__main__":
    collect_all_fundamentals()
