# Stock Predictor AI

An explainable stock prediction platform. It combines price indicators with news sentiment to forecast stock movements, and shows the reasons behind every prediction.

**Demo:** [stock-predictor-ai-kappa.vercel.app](https://stock-predictor-ai-kappa.vercel.app/) (a static preview built from a dated snapshot)
![Dashboard](docs/screenshots/dashboard.png)

## Features

- Forecasts for 10 tickers across four horizons (1 day, 1 week, 1 month and 1 year), powered by 80 trained models
- 20+ engineered features, including technical indicators (RSI, MACD, Bollinger Bands) and news sentiment
- Two sentiment pipelines, using news from Alpha Vantage, GNews and the FNSPID dataset
- SHAP explanations that show which inputs drove each forecast
- React dashboard, FastAPI backend and PostgreSQL storage
- Scheduled data collection and model retraining scripts

## Tech stack

| Layer | Technology |
| --- | --- |
| Frontend | React |
| Backend | Python, FastAPI |
| Database | PostgreSQL (Neon) |
| Machine learning | Python, SHAP explainability |
| Hosting | Vercel (dashboard) |

## How it works

```
Price and news sources
        |
   data_pipeline  -->  PostgreSQL
                          |
        ml (features, training, SHAP explanations)
                          |
                  FastAPI backend
                          |
                  React dashboard
```

## Project structure

```
api/            FastAPI backend and prediction endpoints
config/         Application configuration
data_pipeline/  Price and news collection and processing
database/       Database models and schema
frontend/       React dashboard
ml/             Feature engineering, model training and inference
scripts/        Helper and maintenance scripts
tests/          Automated tests
scheduler.py    Scheduled data collection
requirements.txt  Python dependencies
```

## Branches

- `main` contains the full application source.
- `portfolio-preview` contains the static dashboard preview used for portfolio hosting. It uses a dated snapshot and does not provide live predictions, scheduled data collection or retraining.

## Live preview vs full application

The demo link above is the static preview. It shows the dashboard and a snapshot of predictions, but it is not connected to a running API, so it does not update or generate new forecasts.

Running the full application needs three things that are intentionally not stored in this repository:

1. A PostgreSQL database
2. Credentials for the news and data services, set as environment variables
3. The trained model files, provided separately

## Run locally

Requirements: Python 3, Node.js and a PostgreSQL database.

```bash

# Backend
python -m venv venv
venv\Scripts\activate        # Windows (use: source venv/bin/activate on macOS or Linux)
pip install -r requirements.txt
```

1. Create a `.env` file with your database connection string and API keys. The exact variable names are read in `config/`. Never commit this file.
2. Place the trained model files where the configuration expects them.
3. Start the FastAPI app from `api/` with `uvicorn`.
4. For the dashboard, open the `frontend/` folder, run `npm install`, then use the scripts listed in `frontend/package.json`.

## Security

Do not commit `.env` files, database connection strings, API keys, or private model and data artifacts. The repository ignores these by default.

## Disclaimer

This is an educational and portfolio project. Its predictions are not financial advice and should not be used to make investment decisions.

## Author

Chimaobi Eustace Ndu · [GitHub](https://github.com/ndueustace4) · [LinkedIn](https://www.linkedin.com/in/chimaobi-ndu-163299370)
