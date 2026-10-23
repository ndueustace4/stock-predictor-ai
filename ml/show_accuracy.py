# ml/show_accuracy.py
# ─────────────────────────────────────────────
# Displays two accuracy tables in the terminal:
#   1. Training metrics (from ml/models/training_metrics.json)
#   2. Live resolved predictions (from the predictions DB table)
#
# Usage:
#   py -3 ml/show_accuracy.py
# ─────────────────────────────────────────────

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_pipeline.db_connection import get_connection

# ── Colours ───────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def colour(val, low=50, mid=60):
    if val is None: return f"{'N/A':>7}"
    s = f"{val:>6.1f}%"
    if val >= mid:   return f"{GREEN}{s}{RESET}"
    if val >= low:   return f"{YELLOW}{s}{RESET}"
    return f"{RED}{s}{RESET}"

def bar(val, width=10):
    if val is None: return "-" * width
    filled = int(round(val / 100 * width))
    colour_code = GREEN if val >= 60 else (YELLOW if val >= 50 else RED)
    return colour_code + "#" * filled + RESET + "." * (width - filled)


# ═══════════════════════════════════════════════
# TABLE 1 — Training metrics (test-set)
# ═══════════════════════════════════════════════

metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "models", "training_metrics.json")

print()
print(BOLD + "=" * 90 + RESET)
print(BOLD + "  TABLE 1 — MODEL TRAINING ACCURACY  (measured on held-out test data the model never saw)" + RESET)
print(BOLD + "=" * 90 + RESET)

if not os.path.exists(metrics_path):
    print("  [!] training_metrics.json not found — run: py -3 ml/eval_models.py")
else:
    with open(metrics_path) as f:
        data = json.load(f)

    horizons = data.get("horizons", {})
    horizon_labels = {"1d": "1 Day", "5d": "1 Week", "21d": "1 Month", "252d": "1 Year"}

    for h_key, h_lbl in horizon_labels.items():
        h_data = horizons.get(h_key, {})
        ns_rows = {r["ticker"]: r for r in h_data.get("no_sentiment", [])}
        ws_rows = {r["ticker"]: r for r in h_data.get("with_sentiment", [])}
        if not ns_rows:
            continue

        print()
        print(BOLD + f"  ── HORIZON: {h_lbl} ({h_key}) ──" + RESET)
        print(f"  {'Ticker':<7}  {'─── Without Sentiment ───':^35}  {'─── With Sentiment ───':^35}")
        print(f"  {'':7}  {'Accuracy':>9}  {'F1':>7}  {'Precision':>10}  {'Recall':>8}"
              f"  {'Accuracy':>9}  {'F1':>7}  {'Precision':>10}  {'Recall':>8}")
        print("  " + "-" * 85)

        tickers = sorted(set(ns_rows) | set(ws_rows))
        ns_acc_all, ws_acc_all = [], []
        ns_f1_all,  ws_f1_all  = [], []

        for t in tickers:
            ns = ns_rows.get(t)
            ws = ws_rows.get(t)
            ns_acc = ns["accuracy"] * 100 if ns else None
            ns_f1  = ns["f1"]      * 100 if ns else None
            ns_pre = ns.get("precision", 0) * 100 if ns else None
            ns_rec = ns.get("recall",    0) * 100 if ns else None
            ws_acc = ws["accuracy"] * 100 if ws else None
            ws_f1  = ws["f1"]      * 100 if ws else None
            ws_pre = ws.get("precision", 0) * 100 if ws else None
            ws_rec = ws.get("recall",    0) * 100 if ws else None

            if ns_acc: ns_acc_all.append(ns_acc)
            if ws_acc: ws_acc_all.append(ws_acc)
            if ns_f1:  ns_f1_all.append(ns_f1)
            if ws_f1:  ws_f1_all.append(ws_f1)

            print(f"  {t:<7}  {colour(ns_acc)}  {colour(ns_f1)}  {colour(ns_pre):>10}  {colour(ns_rec):>8}"
                  f"  {colour(ws_acc)}  {colour(ws_f1)}  {colour(ws_pre):>10}  {colour(ws_rec):>8}")

        if ns_acc_all:
            avg_ns_acc = sum(ns_acc_all) / len(ns_acc_all)
            avg_ws_acc = sum(ws_acc_all) / len(ws_acc_all) if ws_acc_all else None
            avg_ns_f1  = sum(ns_f1_all)  / len(ns_f1_all)
            avg_ws_f1  = sum(ws_f1_all)  / len(ws_f1_all) if ws_f1_all else None
            print("  " + "-" * 85)
            print(f"  {'AVERAGE':<7}  {colour(avg_ns_acc)}  {colour(avg_ns_f1)}  {'':>10}  {'':>8}"
                  f"  {colour(avg_ws_acc)}  {colour(avg_ws_f1)}")


# ═══════════════════════════════════════════════
# TABLE 2 — Live resolved predictions
# ═══════════════════════════════════════════════

print()
print(BOLD + "=" * 90 + RESET)
print(BOLD + "  TABLE 2 — LIVE RESOLVED PREDICTIONS  (real predictions checked against actual prices)" + RESET)
print(BOLD + "=" * 90 + RESET)

try:
    conn = get_connection()
    with conn.cursor() as cur:

        # Per ticker per horizon
        cur.execute("""
            SELECT ticker, horizon,
                COUNT(*)                                                      AS total,
                SUM(CASE WHEN correct = true  THEN 1 ELSE 0 END)             AS correct,
                SUM(CASE WHEN correct = false THEN 1 ELSE 0 END)             AS wrong,
                ROUND(AVG(CASE WHEN correct = true THEN 1.0 ELSE 0 END)*100,1) AS acc,
                MIN(prediction_date)  AS from_date,
                MAX(prediction_date)  AS to_date
            FROM predictions
            WHERE actual_label IS NOT NULL
            GROUP BY ticker, horizon
            ORDER BY horizon, ticker
        """)
        rows = cur.fetchall()

        # Overall per horizon
        cur.execute("""
            SELECT horizon,
                COUNT(*) AS total,
                SUM(CASE WHEN correct = true THEN 1 ELSE 0 END) AS correct,
                ROUND(AVG(CASE WHEN correct = true THEN 1.0 ELSE 0 END)*100,1) AS acc
            FROM predictions
            WHERE actual_label IS NOT NULL
            GROUP BY horizon ORDER BY horizon
        """)
        overall = {r[0]: r for r in cur.fetchall()}

        # By direction
        cur.execute("""
            SELECT horizon, predicted_label,
                COUNT(*) AS total,
                ROUND(AVG(CASE WHEN correct = true THEN 1.0 ELSE 0 END)*100,1) AS acc
            FROM predictions
            WHERE actual_label IS NOT NULL
            GROUP BY horizon, predicted_label ORDER BY horizon, predicted_label
        """)
        by_dir = cur.fetchall()

    conn.close()

    if not rows:
        print()
        print("  No resolved predictions found yet.")
    else:
        # Group by horizon
        from collections import defaultdict
        by_horizon = defaultdict(list)
        for r in rows:
            by_horizon[r[1]].append(r)

        h_labels = {"1d": "1 Day", "5d": "1 Week", "21d": "1 Month", "252d": "1 Year"}

        for h_key in ["1d", "5d", "21d", "252d"]:
            h_rows = by_horizon.get(h_key, [])
            ov = overall.get(h_key)

            if not h_rows:
                print()
                print(BOLD + f"  ── HORIZON: {h_labels.get(h_key, h_key)} ({h_key}) ──" + RESET)
                print(f"  [pending — no resolved predictions yet]")
                continue

            from_dt = min(r[6] for r in h_rows)
            to_dt   = max(r[7] for r in h_rows)

            print()
            print(BOLD + f"  ── HORIZON: {h_labels.get(h_key, h_key)} ({h_key}) "
                  f"│ Period: {from_dt} → {to_dt} ──" + RESET)
            print(f"  {'Ticker':<7}  {'Total':>6}  {'Correct':>8}  {'Wrong':>6}  {'Accuracy':>9}  {'Bar (100%)':<12}")
            print("  " + "-" * 60)

            for r in h_rows:
                acc_val = float(r[5]) if r[5] is not None else None
                print(f"  {r[0]:<7}  {r[2]:>6}  {r[3]:>8}  {r[4]:>6}  "
                      f"{colour(acc_val)}  {bar(acc_val)}")

            if ov:
                ov_acc = float(ov[3]) if ov[3] is not None else None
                print("  " + "-" * 60)
                print(f"  {'TOTAL':<7}  {ov[1]:>6}  {ov[2]:>8}  {ov[1]-ov[2]:>6}  "
                      f"{colour(ov_acc)}  {bar(ov_acc)}")

        # Direction breakdown
        print()
        print(BOLD + "  ── BY PREDICTED DIRECTION ──" + RESET)
        print(f"  {'Horizon':<8}  {'Direction':<8}  {'Called':>7}  {'Correct':>8}  {'Accuracy':>9}")
        print("  " + "-" * 48)
        for r in by_dir:
            direction = "UP  " if r[1] == 1 else "DOWN"
            acc_val = float(r[3]) if r[3] is not None else None
            print(f"  {r[0]:<8}  {direction:<8}  {r[2]:>7}  {'':>8}  {colour(acc_val)}")

except Exception as e:
    print(f"  [!] Could not connect to database: {e}")


print()
print(BOLD + "=" * 90 + RESET)
print(f"  {GREEN}Green{RESET} = 60%+   {YELLOW}Yellow{RESET} = 50-60%   {RED}Red{RESET} = below 50%   "
      f"Baseline (random guessing) = 50%")
print(BOLD + "=" * 90 + RESET)
print()
