# data_pipeline/collect_macro.py
# ─────────────────────────────────────────────
# Fetches US Treasury yield data from the FRED API
# and stores it in the `macro_rates` table.
#
# Series fetched:
#   DGS10 — 10-year Treasury yield
#   DGS2  — 2-year Treasury yield
#   yield_curve = DGS10 - DGS2  (negative = inverted = recession signal)
#
# Requires: pip install fredapi
# API key: set FRED_KEY in config/settings.py
# ─────────────────────────────────────────────

import sys
import os
import pandas as pd
from psycopg2.extras import execute_values

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import FRED_KEY, START_DATE
from data_pipeline.db_connection import get_connection


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS macro_rates (
    date         DATE  PRIMARY KEY,
    treasury_10y FLOAT,
    treasury_2y  FLOAT,
    yield_curve  FLOAT
);
"""


def fetch_fred(series_id: str, start: str) -> pd.Series:
    try:
        from fredapi import Fred
    except ImportError:
        print("[ERROR] fredapi not installed. Run: pip install fredapi")
        raise

    fred = Fred(api_key=FRED_KEY)
    try:
        data = fred.get_series(series_id, observation_start=start)
        print(f"  {series_id}: {len(data)} observations ({data.index[0].date()} to {data.index[-1].date()})")
        return data
    except Exception as e:
        print(f"  [ERROR] FRED fetch failed for {series_id}: {e}")
        return pd.Series(dtype=float)


def collect_macro_rates():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()

    print("Fetching Treasury yields from FRED...")
    t10y = fetch_fred("DGS10", START_DATE)
    t2y  = fetch_fred("DGS2",  START_DATE)

    if t10y.empty and t2y.empty:
        print("[ERROR] No data fetched — check FRED_KEY in config/settings.py")
        conn.close()
        return

    df = pd.DataFrame({"treasury_10y": t10y, "treasury_2y": t2y})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # FRED has NaN on weekends/holidays — forward-fill so every calendar day has a value
    df = df.ffill()
    df["yield_curve"] = df["treasury_10y"] - df["treasury_2y"]
    df = df.dropna(subset=["treasury_10y", "treasury_2y"])

    def _f(v):
        return float(v) if pd.notna(v) else None

    rows = [
        (idx.date(), _f(row["treasury_10y"]), _f(row["treasury_2y"]), _f(row["yield_curve"]))
        for idx, row in df.iterrows()
    ]

    sql = """
        INSERT INTO macro_rates (date, treasury_10y, treasury_2y, yield_curve)
        VALUES %s
        ON CONFLICT (date) DO UPDATE SET
            treasury_10y = EXCLUDED.treasury_10y,
            treasury_2y  = EXCLUDED.treasury_2y,
            yield_curve  = EXCLUDED.yield_curve
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    conn.close()

    print(f"Done. {len(rows)} rows upserted ({df.index.min().date()} to {df.index.max().date()})")


if __name__ == "__main__":
    collect_macro_rates()
