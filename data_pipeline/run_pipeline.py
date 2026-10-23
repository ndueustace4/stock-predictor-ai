# data_pipeline/run_pipeline.py
# ─────────────────────────────────────────────
# Master script — runs all pipeline phases.
# Usage:
#   python data_pipeline/run_pipeline.py            # full pipeline
#   python data_pipeline/run_pipeline.py --backfill # Alpha backfill only
#   python data_pipeline/run_pipeline.py --gnews    # GNews only
#   python data_pipeline/run_pipeline.py --monitor  # monitoring report only
# ─────────────────────────────────────────────

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.db_connection import test_connection
from data_pipeline.collect_stocks import collect_all_stocks
from data_pipeline.collect_news import collect_all_news
from data_pipeline.collect_alpha import run_alpha_backfill
from data_pipeline.sentiment_analysis import run_sentiment_analysis, show_sample_results
from data_pipeline.feature_engineering import run_feature_engineering, show_feature_sample
from data_pipeline.monitoring import daily_summary
from ml.train_model import run_training


def run_full():
    print("\n📡 Stock Predictor — Full Pipeline\n")

    print("Step 1: Testing database connection...")
    test_connection()

    print("\nStep 2: Collecting stock price data...")
    collect_all_stocks()

    print("\nStep 3: GNews live articles...")
    collect_all_news()

    print("\nStep 4: Alpha Vantage backfill (one batch)...")
    run_alpha_backfill(max_windows_per_run=3)

    print("\nStep 5: VADER sentiment (catch any unscored rows)...")
    run_sentiment_analysis()
    show_sample_results()

    print("\nStep 6: Feature engineering...")
    run_feature_engineering()
    show_feature_sample()

    print("\nStep 7: Train ML models...")
    run_training()

    print("\nStep 8: Pipeline monitoring report...")
    daily_summary()

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--full"

    if mode == "--backfill":
        print("\n🔵 Alpha Vantage backfill run\n")
        test_connection()
        run_alpha_backfill(max_windows_per_run=50)
        daily_summary()

    elif mode == "--gnews":
        print("\n🟢 GNews live collection run\n")
        test_connection()
        collect_all_news()

    elif mode == "--monitor":
        daily_summary()

    elif mode == "--sentiment":
        print("\n🔍 Sentiment re-score (unscored rows only)\n")
        test_connection()
        run_sentiment_analysis()
        show_sample_results()

    else:
        run_full()
