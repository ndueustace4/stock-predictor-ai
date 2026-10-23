import { useEffect, useState } from "react";
import { getApiData } from "../api";

const HORIZONS = [
  { label: "1 Day",   key: "1d",   note: "Predicts next trading day" },
  { label: "1 Week",  key: "5d",   note: "Predicts 5 trading days ahead" },
  { label: "1 Month", key: "21d",  note: "Predicts 21 trading days ahead" },
  { label: "1 Year",  key: "252d", note: "Predicts 252 trading days ahead" },
];

export default function PredictionPanel({ ticker, useSentiment, horizon, setHorizon }) {
  const [data, setData]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    getApiData(`/predict/${ticker}?use_sentiment=${useSentiment}&horizon=${horizon}`)
      .then(result => { setData(result); setLoading(false); })
      .catch(() => { setError("Failed to load prediction"); setLoading(false); });
  }, [ticker, useSentiment, horizon]);

  const note = HORIZONS.find(h => h.key === horizon)?.note;
  const isUp = data?.prediction === "UP";
  const conf = data ? (data.confidence * 100).toFixed(1) : 0;

  return (
    <div className="card">
      <h2>Prediction — {ticker}</h2>
      <div className="horizon-group">
        {HORIZONS.map(h => (
          <button key={h.key} className={`horizon-btn ${horizon===h.key?"active":""}`} onClick={() => setHorizon(h.key)}>{h.label}</button>
        ))}
      </div>
      <p className="horizon-note">{note}</p>

      {loading && <p className="loading">Analyzing market data...</p>}
      {error   && <p className="error">{error}</p>}

      {!loading && !error && data && <>
        <div className="prediction-box">
          <div className="prediction-glow">
            <div className={`prediction-arrow ${isUp?"up":"down"}`}>{isUp?"↑":"↓"}</div>
          </div>
          <div className={`prediction-label ${isUp?"up":"down"}`}>Price {data.prediction}</div>
          <div className="confidence-row">
            <div className="confidence-bar-wrap">
              <div className={`confidence-bar ${isUp?"up":"down"}`} style={{ width: `${conf}%` }} />
            </div>
            <div className="confidence-text">Confidence: <span>{conf}%</span></div>
          </div>
          <div className="model-badge">{useSentiment ? "With Sentiment" : "No Sentiment"}</div>
        </div>

        <div className="factors">
          <p className="factors-title">Top Drivers</p>
          {data.factors.slice(0,4).map((f,i) => (
            <div className="factor" key={i}>
              <span className="factor-name">{f.label}</span>
              <span className={`factor-value ${f.shap_value>=0?"pos":"neg"}`}>
                {f.shap_value>=0?"+":""}{f.shap_value.toFixed(4)}
              </span>
            </div>
          ))}
        </div>
      </>}
    </div>
  );
}
