# scheduler.py
# ─────────────────────────────────────────────
# Unified scheduler for the news pipeline.
#
# Schedule:
#   GNews      — every 30 minutes  (live updates, Essential plan)
#   Stocks     — 20:30 UTC  (fetch prices after US market close)
#   Features   — 20:45 UTC  (rebuild feature table after stock fetch)
#   Monitor    — 21:00 UTC  (daily summary report)
#   Predict    — 22:00 UTC  (log predictions + resolve actuals)
#
# Rate-limit guard:
#   GNews      — max 900 requests/day  (Essential plan = 1000, keep 100 buffer)
#
# Usage:
#   python scheduler.py
#   (keep running; Ctrl+C to stop)
# ─────────────────────────────────────────────

import sys
import os
import logging
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_pipeline.monitoring import requests_today, daily_summary

# ── Logging ───────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/scheduler.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)


# ─────────────────────────────────────────────
# Job wrappers with budget guards
# ─────────────────────────────────────────────

GNEWS_DAILY_LIMIT = 900  # Essential plan = 1000 req/day; keep 100 buffer


def job_gnews():
    """
    GNews live collection job — runs every 30 minutes.
    Aborts if daily request budget is exhausted.
    After collecting news, triggers event-driven prediction update:
    any ticker with today's articles gets its sentiment_today patched
    and predictions re-run immediately (UPSERT).
    """
    used = requests_today("gnews")
    from config.settings import STOCKS
    remaining = GNEWS_DAILY_LIMIT - used

    if remaining <= 0:
        log.warning("GNews daily budget exhausted (%d/%d). Skipping run.", used, GNEWS_DAILY_LIMIT)
        return

    log.info("GNews job starting. Budget: %d/%d used.", used, GNEWS_DAILY_LIMIT)

    try:
        from data_pipeline.collect_news import collect_all_news
        collect_all_news()
        log.info("GNews job complete.")
    except Exception as e:
        log.error("GNews job FAILED: %s", e, exc_info=True)
        return

    # Event trigger: re-predict any ticker whose sentiment shifted today
    try:
        from ml.event_predict import trigger_event_predictions
        trigger_event_predictions()
    except Exception as e:
        log.error("Event prediction trigger FAILED: %s", e, exc_info=True)



def job_stocks():
    """Fetch latest stock prices after US market close — runs at 20:30 UTC."""
    log.info("Stock price fetch starting...")
    try:
        from data_pipeline.collect_stocks import collect_all_stocks
        collect_all_stocks()
        log.info("Stock price fetch complete.")
    except Exception as e:
        log.error("Stock fetch job FAILED: %s", e, exc_info=True)


def job_features():
    """Rebuild feature table after stock fetch — runs at 20:45 UTC."""
    log.info("Feature engineering starting...")
    try:
        from data_pipeline.feature_engineering import run_feature_engineering
        run_feature_engineering()
        log.info("Feature engineering complete.")
    except Exception as e:
        log.error("Feature engineering job FAILED: %s", e, exc_info=True)


def job_predict():
    """Log daily predictions + resolve matured ones — runs at 22:00."""
    log.info("Prediction job starting...")
    try:
        from ml.predict import run_predictions, resolve_actuals
        from data_pipeline.db_connection import get_connection
        conn = get_connection()
        run_predictions(conn)
        resolve_actuals(conn)
        conn.close()
        log.info("Prediction job complete.")
    except Exception as e:
        log.error("Prediction job FAILED: %s", e, exc_info=True)


def job_retrain():
    """Retrain all models with latest data — runs at 23:00 UTC daily."""
    log.info("Retrain job starting...")
    try:
        from ml.train_model import run_training
        run_training()
        log.info("Retrain job complete.")
    except Exception as e:
        log.error("Retrain job FAILED: %s", e, exc_info=True)


def job_keepalive():
    """Ping Neon DB every 4 minutes to prevent the compute from pausing."""
    try:
        from data_pipeline.db_connection import get_connection
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
    except Exception as e:
        log.warning("Keep-alive ping failed: %s", e)


def job_monitor():
    """Daily summary report — runs at 21:00."""
    log.info("Generating daily monitoring report...")
    try:
        daily_summary()
    except Exception as e:
        log.error("Monitor job FAILED: %s", e, exc_info=True)


# ─────────────────────────────────────────────
# Scheduler setup
# ─────────────────────────────────────────────

def build_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone="UTC")

    # Neon keep-alive: every 4 minutes (Neon pauses after ~5 min idle)
    scheduler.add_job(
        job_keepalive,
        trigger=IntervalTrigger(minutes=4),
        id="neon_keepalive",
        name="Neon DB Keep-Alive",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )

    # GNews: every 30 minutes
    scheduler.add_job(
        job_gnews,
        trigger=IntervalTrigger(minutes=30),
        id="gnews_live",
        name="GNews Live Collector",
        max_instances=1,        # don't overlap
        coalesce=True,          # if we miss a run, fire once not many times
        misfire_grace_time=120, # seconds before a missed run is skipped
    )

    # Stock prices at 20:30 UTC (US market closes 20:00 UTC)
    scheduler.add_job(
        job_stocks,
        trigger=CronTrigger(hour=20, minute=30),
        id="daily_stocks",
        name="Daily Stock Price Fetch",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )

    # Feature engineering at 20:45 UTC (after stock fetch)
    scheduler.add_job(
        job_features,
        trigger=CronTrigger(hour=20, minute=45),
        id="daily_features",
        name="Daily Feature Engineering",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )

    # Daily monitor report at 21:00 UTC
    scheduler.add_job(
        job_monitor,
        trigger=CronTrigger(hour=21, minute=0),
        id="daily_monitor",
        name="Daily Monitor Report",
        max_instances=1,
        coalesce=True,
    )

    # Daily predictions at 22:00 UTC (after US market close + feature update)
    scheduler.add_job(
        job_predict,
        trigger=CronTrigger(hour=22, minute=0),
        id="daily_predict",
        name="Daily Predictions",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )

    # Daily retrain at 23:00 UTC (after predictions, uses latest prices + news)
    scheduler.add_job(
        job_retrain,
        trigger=CronTrigger(hour=23, minute=0),
        id="daily_retrain",
        name="Daily Model Retrain",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )

    return scheduler


def main():
    log.info("=" * 55)
    log.info("Stock Predictor Scheduler starting")
    log.info("GNews  : every 30 minutes (budget: %d req/day, Essential plan)", GNEWS_DAILY_LIMIT)
    log.info("Stocks : daily at 20:30 UTC")
    log.info("Features: daily at 20:45 UTC")
    log.info("Monitor: daily at 21:00 UTC")
    log.info("Predict: daily at 22:00 UTC")
    log.info("Retrain: daily at 23:00 UTC")
    log.info("=" * 55)

    scheduler = build_scheduler()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped by user.")


if __name__ == "__main__":
    main()
