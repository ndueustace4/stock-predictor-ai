# Stock Predictor AI

This repository contains the stock prediction project source: a React dashboard,
a FastAPI backend, PostgreSQL data pipeline, model training and inference code,
and scheduled collection scripts.

## Branches

- `main` contains the full application source.
- `portfolio-preview` contains the static dashboard preview for portfolio
  hosting. It uses a dated snapshot and does not provide live predictions,
  scheduled data collection, or retraining.

## Important runtime files

The repository intentionally does not include `.env` credentials or trained
model binaries. Configure database and service credentials through environment
variables, and provide the model artifacts separately when running the live
API. The static portfolio preview does not need database credentials or model
files after its snapshot has been generated.

Do not commit `.env` files, database connection strings, API keys, or private
model/data artifacts.

This static portfolio preview is deployed from the portfolio-preview branch.
