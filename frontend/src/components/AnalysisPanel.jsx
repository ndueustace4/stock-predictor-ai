// frontend/src/components/AnalysisPanel.jsx
// Auto-generated plain English analysis for the selected ticker

import { useEffect, useState } from "react";
import { getApiData } from "../api";

export default function AnalysisPanel({ ticker, horizon }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    setData(null);
    setError(null);

    getApiData(`/analysis/${ticker}?horizon=${horizon}`)
      .then(result => {
        setData(result);
        setLoading(false);
      })
      .catch(() => {
        setError("Could not load analysis.");
        setLoading(false);
      });
  }, [ticker, horizon]);

  const directionColor = data?.prediction === "UP" ? "#4ade80" : "#f87171";
  const directionArrow = data?.prediction === "UP" ? "↑" : "↓";

  return (
    <div className="card analysis-panel">
      <div className="analysis-header">
        <span className="analysis-title">{data?.demo_snapshot ? "Snapshot analysis" : "Analysis"}</span>
        {data && (
          <span className="analysis-badge" style={{ color: directionColor }}>
            {directionArrow} {data.prediction} · {data.confidence}% confidence
          </span>
        )}
      </div>

      {loading && (
        <div className="analysis-loading">
          <div className="analysis-skeleton" />
          <div className="analysis-skeleton short" />
          <div className="analysis-skeleton" />
          <div className="analysis-skeleton short" />
          <p className="analysis-loading-text">Generating analysis...</p>
        </div>
      )}

      {error && (
        <p className="analysis-error">{error}</p>
      )}

      {data && !loading && (
        <div className="analysis-sections">

          <div className="analysis-section">
            <div className="analysis-section-label">
              Main Driver
            </div>
            <p className="analysis-text">{data.analysis.main_reason}</p>
          </div>

          <div className="analysis-section">
            <div className="analysis-section-label">
              News Impact
            </div>
            <p className="analysis-text">{data.analysis.news_impact}</p>
          </div>

          <div className="analysis-section">
            <div className="analysis-section-label">
              Why {data.confidence}% Confidence?
            </div>
            <p className="analysis-text">{data.analysis.confidence_reason}</p>
          </div>

          <p className="analysis-hint">
            Want more detail? Ask the AI Chat — "latest news on {ticker}" or "explain the RSI for {ticker}"
          </p>

        </div>
      )}
    </div>
  );
}
