# data_pipeline/db_connection.py

import psycopg2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_CONFIG


def get_connection():
    """Return a live psycopg2 connection to stock_predictor DB."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.OperationalError as e:
        print(f"[ERROR] Could not connect to PostgreSQL: {e}")
        print("Check that PostgreSQL is running and your password in config/settings.py is correct.")
        raise


def test_connection():
    """Quick sanity check — prints success or error."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"[OK] Connected to PostgreSQL: {version}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[FAIL] {e}")


if __name__ == "__main__":
    test_connection()