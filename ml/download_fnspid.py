# ml/download_fnspid.py
# Streams FNSPID All_external.csv from HuggingFace and saves only our 10 tickers.
# Usage: python ml/download_fnspid.py

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import STOCKS

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATASET     = "Zihan1004/FNSPID"
DATA_FILE   = "Stock_news/All_external.csv"
OUT_PATH    = "ml/fnspid_filtered.csv"
CHUNK_SIZE  = 50_000

TICKERS = set(STOCKS) | {"FB"}  # META was FB before Oct 2021


def stream_and_filter():
    from huggingface_hub import hf_hub_download

    print("Downloading All_external.csv from HuggingFace (streaming by chunks)...")
    print(f"Filtering for: {sorted(TICKERS)}")

    local_path = hf_hub_download(
        repo_id=DATASET,
        filename=DATA_FILE,
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir="ml/fnspid_raw",
    )

    print(f"File cached at: {local_path}")
    print("Filtering rows for our tickers...")

    filtered_chunks = []
    total_rows = 0
    kept_rows  = 0

    for chunk in pd.read_csv(local_path, chunksize=CHUNK_SIZE, low_memory=False):
        total_rows += len(chunk)

        ticker_col = "Stock_symbol"

        mask = chunk[ticker_col].isin(TICKERS)
        filtered = chunk[mask]
        kept_rows += len(filtered)
        if not filtered.empty:
            filtered_chunks.append(filtered)

        if total_rows % 500_000 == 0:
            print(f"  Scanned {total_rows:,} rows, kept {kept_rows:,}...")

    if not filtered_chunks:
        print("No matching rows found. Check ticker column name.")
        return

    result = pd.concat(filtered_chunks, ignore_index=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\nDone! Scanned {total_rows:,} rows, kept {kept_rows:,} rows.")
    print(f"Saved -> {OUT_PATH}")
    print(f"\nColumns: {list(result.columns)}")
    print(f"\nSample:\n{result.head()}")
    print(f"\nTicker counts:\n{result[ticker_col].value_counts()}")


if __name__ == "__main__":
    stream_and_filter()
