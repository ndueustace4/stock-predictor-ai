import { useState, useEffect } from "react";
import StockSelector from "./components/StockSelector";
import PriceChart from "./components/PriceChart";
import PredictionPanel from "./components/PredictionPanel";
import SentimentPanel from "./components/SentimentPanel";
import ExplanationPanel from "./components/ExplanationPanel";
import AnalysisPanel from "./components/AnalysisPanel";
import ChatPanel from "./components/ChatPanel";
import { isStaticPreview, loadDemoSnapshot } from "./api";
import "./App.css";

const TICKERS = ["AAPL","TSLA","MSFT","GOOGL","AMZN","NVDA","META","NFLX","AMD","BABA"];

export default function App() {
  const [ticker, setTicker]             = useState("AAPL");
  const [useSentiment, setUseSentiment] = useState(true);
  const [horizon, setHorizon]           = useState("1d");
  const [chatOpen, setChatOpen]         = useState(false);
  const [theme, setTheme]               = useState(() => localStorage.getItem("theme") || "dark");
  const [snapshotMeta, setSnapshotMeta] = useState(null);

  useEffect(() => {
    if (isStaticPreview) {
      loadDemoSnapshot()
        .then(snapshot => setSnapshotMeta(snapshot.metadata))
        .catch(error => console.error("Could not load portfolio snapshot:", error));
    }
  }, []);

  useEffect(() => {
    document.body.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === "dark" ? "light" : "dark");

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="header-logo">Stock<span>Predictor</span></div>
          <span className="header-badge">{isStaticPreview ? "PORTFOLIO PREVIEW" : "AI POWERED"}</span>
        </div>
        <div className="header-right">
          <StockSelector tickers={TICKERS} selected={ticker} onChange={setTicker} />
          <label className={`sentiment-toggle ${useSentiment ? "active" : ""}`}>
            <input type="checkbox" checked={useSentiment} onChange={e => setUseSentiment(e.target.checked)} />
            Sentiment Model
          </label>
          <button className="theme-toggle-btn" onClick={toggleTheme} title="Toggle light/dark mode">
            {theme === "dark" ? "☀ Light" : "☾ Dark"}
          </button>
          <button
            className="chat-toggle-btn"
            onClick={() => setChatOpen(true)}
            disabled={isStaticPreview}
            title={isStaticPreview ? "AI chat requires a live API service" : "Open AI chat"}
          >
            <span className="chat-btn-icon">◈</span> {isStaticPreview ? "AI Chat unavailable" : "AI Chat"}
          </button>
        </div>
      </header>

      {isStaticPreview && (
        <div className="preview-banner" role="status">
          Static portfolio demo
          {snapshotMeta?.price_as_of && <> · market data through {snapshotMeta.price_as_of}</>}
          {snapshotMeta?.news_as_of && <> · saved news through {snapshotMeta.news_as_of}</>}
          {" · "}Predictions are model outputs for demonstration, not financial advice.
        </div>
      )}

      <div className="layout">
        <div className="area-price">
          <div className="card"><PriceChart ticker={ticker} /></div>
        </div>
        <div className="area-sidebar">
          <PredictionPanel ticker={ticker} useSentiment={useSentiment} horizon={horizon} setHorizon={setHorizon} />
          <SentimentPanel ticker={ticker} />
        </div>
        <div className="area-explain">
          <AnalysisPanel ticker={ticker} horizon={horizon} />
          <div className="card"><ExplanationPanel ticker={ticker} useSentiment={useSentiment} horizon={horizon} /></div>
        </div>
      </div>

      <footer className="footer">
        StockPredictor AI — XGBoost · SHAP · VADER NLP · React &nbsp;|&nbsp; Chimaobi Eustace Ndu
      </footer>

      <ChatPanel
        isOpen={chatOpen}
        onClose={() => setChatOpen(false)}
        ticker={ticker}
      />
    </div>
  );
}
