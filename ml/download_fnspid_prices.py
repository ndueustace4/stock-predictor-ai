# ml/download_fnspid_prices.py
# Downloads FNSPID stock price data and extracts our 10 tickers.

import os
import sys
import zipfile
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import STOCKS

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
DATASET   = "Zihan1004/FNSPID"
DATA_FILE = "Stock_price/full_history.zip"
OUT_DIR   = "ml/fnspid_prices"

TICKERS = set(STOCKS) | {"FB"}

def main():
    from huggingface_hub import hf_hub_download
    import pandas as pd

    print("Downloading full_history.zip...")
    local_path = hf_hub_download(
        repo_id=DATASET,
        filename=DATA_FILE,
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir="ml/fnspid_raw",
    )
    print(f"Downloaded: {local_path}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Extracting and checking contents...")
    with zipfile.ZipFile(local_path, "r") as z:
        names = z.namelist()
        print(f"Files in zip: {len(names)}")
        print(f"Sample: {names[:10]}")

        found = []
        for name in names:
            # Try to match ticker from filename
            base = os.path.basename(name).replace(".pkl", "").replace(".csv", "").upper()
            if base in TICKERS:
                found.append((name, base))

        print(f"\nMatched tickers: {[t for _, t in found]}")

        for name, ticker in found:
            data = z.read(name)
            out_path = os.path.join(OUT_DIR, f"{ticker}.pkl")
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"  Saved {ticker} -> {out_path}")

    # Load one to inspect schema
    if found:
        sample_ticker = found[0][1]
        with open(os.path.join(OUT_DIR, f"{sample_ticker}.pkl"), "rb") as f:
            df = pickle.load(f)
        print(f"\nSample ({sample_ticker}) schema:")
        print(df.dtypes)
        print(df.head())

if __name__ == "__main__":
    main()
