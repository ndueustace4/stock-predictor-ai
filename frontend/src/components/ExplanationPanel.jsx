import { useEffect, useState } from "react";
import { getApiData } from "../api";

function SHAPBar({ label, value, maxVal }) {
  const pct = Math.min((Math.abs(value)/maxVal)*100, 100);
  const pos = value >= 0;
  return (
    <div className="shap-bar">
      <span className="shap-label">{label}</span>
      <div className="shap-track"><div className={`shap-fill ${pos?"pos":"neg"}`} style={{width:`${pct}%`}} /></div>
      <span className={`shap-val ${pos?"pos":"neg"}`}>{pos?"+":""}{value.toFixed(4)}</span>
    </div>
  );
}

export default function ExplanationPanel({ ticker, useSentiment, horizon }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    getApiData(`/compare/${ticker}?horizon=${horizon}`)
      .then(result => { setData(result); setLoading(false); })
      .catch(() => { setError("Failed to load explanation"); setLoading(false); });
  }, [ticker, horizon]);

  if (loading) return <div><h2>AI Explanation (SHAP)</h2><p className="loading">Computing SHAP values...</p></div>;
  if (error)   return <div><h2>AI Explanation (SHAP)</h2><p className="error">{error}</p></div>;

  const noSent   = data.model_without_sentiment;
  const withSent = data.model_with_sentiment;
  const allVals  = [...(noSent.top_factors||[]), ...(withSent.top_factors||[])].map(f=>Math.abs(f.shap_value));
  const maxVal   = Math.max(...allVals, 0.01);

  return (
    <div>
      <h2>Why did the model predict this? — SHAP Explainability</h2>
      <div className="explanation-grid">
        <div className="explanation-box">
          <h3>🔵 Without Sentiment</h3>
          {(noSent.top_factors||[]).map((f,i) => <SHAPBar key={i} label={f.label} value={f.shap_value} maxVal={maxVal} />)}
          <div className="plain-english">{noSent.plain_english}</div>
        </div>
        <div className="explanation-box">
          <h3>🟠 With Sentiment</h3>
          {(withSent.top_factors||[]).map((f,i) => <SHAPBar key={i} label={f.label} value={f.shap_value} maxVal={maxVal} />)}
          <div className="plain-english">{withSent.plain_english}</div>
        </div>
      </div>
    </div>
  );
}
