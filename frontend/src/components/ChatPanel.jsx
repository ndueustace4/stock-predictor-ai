// frontend/src/components/ChatPanel.jsx
// ─────────────────────────────────────────────
// ChatGPT-style slide-in panel for stock Q&A.
// Sends messages to POST /chat, injects live DB
// data via Claude on the backend.
// ─────────────────────────────────────────────

import { useState, useRef, useEffect } from "react";
import axios from "axios";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const QUICK_QUESTIONS = [
  "Which stock looks most bullish right now?",
  "Give me a sentiment summary for all stocks",
  "Which stocks are oversold right now?",
  "Compare TSLA and NVDA predictions",
];

const GREETING = "Hi! Ask me anything about your stocks — predictions, sentiment, technicals, or why the model made a call.";

// Lightweight markdown renderer — handles bold, bullets, line breaks
function MiniMarkdown({ text }) {
  return (
    <div>
      {text.split("\n").map((line, i) => {
        // Bullet points
        if (line.startsWith("- ") || line.startsWith("• ")) {
          return (
            <div key={i} style={{ display: "flex", gap: 6, marginBottom: 3 }}>
              <span style={{ color: "#ff6622", flexShrink: 0 }}>•</span>
              <span dangerouslySetInnerHTML={{ __html: boldify(line.slice(2)) }} />
            </div>
          );
        }
        // Empty line = spacer
        if (line.trim() === "") return <div key={i} style={{ height: 6 }} />;
        // Normal line
        return (
          <div key={i} style={{ marginBottom: 2 }}
            dangerouslySetInnerHTML={{ __html: boldify(line) }}
          />
        );
      })}
    </div>
  );
}

function boldify(text) {
  // **bold** → <strong>bold</strong>
  return text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

export default function ChatPanel({ isOpen, onClose, ticker }) {
  const [messages, setMessages] = useState([
    { role: "assistant", content: GREETING },
  ]);
  const [input, setInput]     = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef             = useRef(null);
  const inputRef              = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Focus input when panel opens
  useEffect(() => {
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 150);
  }, [isOpen]);

  const sendMessage = async (text) => {
    const userText = (text ?? input).trim();
    if (!userText || loading) return;

    const updated = [...messages, { role: "user", content: userText }];
    setMessages(updated);
    setInput("");
    setLoading(true);

    try {
      // Pass conversation history (skip the initial frontend greeting, last 10 turns)
      const history = updated.slice(1, -1).slice(-10);

      const res = await axios.post(`${API}/chat`, {
        message: userText,
        history,
      });

      setMessages(prev => [
        ...prev,
        { role: "assistant", content: res.data.reply },
      ]);
    } catch (err) {
      const detail = err.response?.data?.detail
        ?? "Could not reach the API server. Is it running?";
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: `⚠️ ${detail}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const showChips = messages.length === 1; // only before first user message

  return (
    <>
      {/* Dim backdrop */}
      <div
        className={`chat-backdrop ${isOpen ? "open" : ""}`}
        onClick={onClose}
      />

      {/* Slide-in panel */}
      <div className={`chat-panel ${isOpen ? "open" : ""}`}>

        {/* ── Header ── */}
        <div className="chat-header">
          <div className="chat-header-left">
            <span className="chat-icon">◈</span>
            <div>
              <div className="chat-title">Stock AI Assistant</div>
              <div className="chat-subtitle">Powered by real-time data</div>
            </div>
          </div>
          <button className="chat-close" onClick={onClose} aria-label="Close">✕</button>
        </div>

        {/* ── Messages ── */}
        <div className="chat-messages">
          {messages.map((msg, i) => (
            <div key={i} className={`chat-bubble ${msg.role}`}>
              {msg.role === "assistant" && (
                <div className="bubble-avatar">AI</div>
              )}
              <div className="bubble-content">
                <MiniMarkdown text={msg.content} />
              </div>
            </div>
          ))}

          {/* Typing indicator */}
          {loading && (
            <div className="chat-bubble assistant">
              <div className="bubble-avatar">AI</div>
              <div className="bubble-content">
                <span className="chat-typing">
                  <span /><span /><span />
                </span>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* ── Quick question chips ── */}
        {showChips && (
          <div className="chat-chips">
            {QUICK_QUESTIONS.map((q, i) => (
              <button
                key={i}
                className="chat-chip"
                onClick={() => sendMessage(q)}
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {/* ── Input row ── */}
        <div className="chat-input-row">
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder={`Ask about ${ticker} or any stock...`}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={loading}
          />
          <button
            className="chat-send"
            onClick={() => sendMessage()}
            disabled={!input.trim() || loading}
            aria-label="Send"
          >
            ➤
          </button>
        </div>

      </div>
    </>
  );
}
