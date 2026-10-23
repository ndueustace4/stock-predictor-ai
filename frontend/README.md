# Stock Predictor portfolio preview

The frontend can run as a static portfolio demo without a hosted API. In that mode, it reads `public/demo-data.json`, a generated snapshot of market data and saved-model predictions. The snapshot includes dated prices, prediction outputs, SHAP factors, and aggregated sentiment; it deliberately excludes individual news headlines and credentials.

## Generate or refresh the demo snapshot

From the repository root, configure a private `.env` with `DATABASE_URL` pointing to the intended PostgreSQL database. Install the Python requirements if needed, then export the snapshot:

```powershell
python scripts\export_demo_snapshot.py
```

This calculates predictions for the configured tickers, four horizons, and both sentiment settings. It writes `frontend/public/demo-data.json`, which is intended to be public in a portfolio deployment. Review the data dates before publishing it.

## Run and build the frontend

```powershell
Set-Location frontend
npm ci
npm run dev
```

For a production build:

```powershell
npm run build
npm run preview
```

Deploy the `frontend` directory to Vercel as a Vite project, with `npm run build` as the build command and `dist` as the output directory. Leave `VITE_API_URL` unset for static preview mode. To use the live API instead, set `VITE_API_URL` to the API base URL; never put secrets in frontend environment variables.

The static preview is not live, does not provide AI chat, and may display older news sentiment than prices. Its banner reports the snapshot dates. Re-export and redeploy the snapshot when you want to refresh the portfolio demo.
