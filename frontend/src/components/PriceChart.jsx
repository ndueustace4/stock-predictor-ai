import { useEffect, useState, useRef, useCallback } from "react";
import { getApiData } from "../api";
import { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

const RANGES = [
  { label: "1M", days: 30 },
  { label: "3M", days: 90 },
  { label: "6M", days: 120 },
  { label: "1Y", days: 252 },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "rgba(10,10,15,0.95)", border: "1px solid rgba(255,85,0,0.2)", borderRadius: 10, padding: "10px 14px", fontSize: 12 }}>
      <p style={{ color: "#6b7280", marginBottom: 6 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, fontWeight: 700 }}>{p.name}: ${p.value?.toLocaleString()}</p>
      ))}
    </div>
  );
};

function CandlestickChart({ data }) {
  const containerRef = useRef(null);
  const [width, setWidth] = useState(600);
  const [hovered, setHovered] = useState(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    setWidth(el.getBoundingClientRect().width);
    const ro = new ResizeObserver(entries => setWidth(entries[0].contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  if (!data.length) return null;

  const height = 260;
  const pad    = { top: 10, right: 10, bottom: 25, left: 58 };
  const innerW = width  - pad.left - pad.right;
  const innerH = height - pad.top  - pad.bottom;

  const priceMax = Math.max(...data.map(d => d.high)) * 1.001;
  const priceMin = Math.min(...data.map(d => d.low))  * 0.999;

  const toY   = p => pad.top  + innerH * (1 - (p - priceMin) / (priceMax - priceMin));
  const toX   = i => pad.left + (i + 0.5) * (innerW / data.length);
  const candW = Math.max(2, innerW / data.length * 0.65);
  const xStep = Math.max(1, Math.floor(data.length / 6));

  const yTicks = Array.from({ length: 5 }, (_, i) =>
    priceMin + (priceMax - priceMin) * i / 4
  );

  const handleMouseMove = useCallback((e) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx  = e.clientX - rect.left - pad.left;
    const idx = Math.round(mx / (innerW / data.length) - 0.5);
    if (idx >= 0 && idx < data.length)
      setHovered({ idx, clientX: e.clientX - rect.left });
    else
      setHovered(null);
  }, [data.length, innerW]);

  return (
    <div ref={containerRef} style={{ width: "100%", position: "relative" }}>
      <svg width={width} height={height}
           onMouseMove={handleMouseMove}
           onMouseLeave={() => setHovered(null)}>

        {/* Grid + Y labels */}
        {yTicks.map((v, i) => (
          <g key={i}>
            <line x1={pad.left} y1={toY(v)} x2={width - pad.right} y2={toY(v)}
                  stroke="rgba(255,255,255,0.04)" strokeWidth={1} />
            <text x={pad.left - 5} y={toY(v) + 4} textAnchor="end" fill="#374151" fontSize={10}>
              ${v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(0)}
            </text>
          </g>
        ))}

        {/* Candlesticks */}
        {data.map((d, i) => {
          const cx    = toX(i);
          const green = d.close >= d.open;
          const color = green ? "#22c55e" : "#ef4444";
          const bTop  = toY(Math.max(d.open, d.close));
          const bBot  = toY(Math.min(d.open, d.close));
          return (
            <g key={i}>
              <line x1={cx} y1={toY(d.high)} x2={cx} y2={toY(d.low)}
                    stroke={color} strokeWidth={1} />
              <rect x={cx - candW / 2} y={bTop} width={candW}
                    height={Math.max(1, bBot - bTop)} fill={color}
                    fillOpacity={hovered?.idx === i ? 1 : 0.8} />
            </g>
          );
        })}

        {/* X labels */}
        {data.map((d, i) => i % xStep === 0 && (
          <text key={i} x={toX(i)} y={height - 5}
                textAnchor="middle" fill="#374151" fontSize={10}>
            {d.date}
          </text>
        ))}

        {/* Crosshair */}
        {hovered && (
          <line x1={toX(hovered.idx)} y1={pad.top}
                x2={toX(hovered.idx)} y2={height - pad.bottom}
                stroke="rgba(255,255,255,0.15)" strokeWidth={1} strokeDasharray="3 3" />
        )}
      </svg>

      {/* OHLC tooltip */}
      {hovered && data[hovered.idx] && (() => {
        const d     = data[hovered.idx];
        const green = d.close >= d.open;
        const onLeft = hovered.clientX > width / 2;
        return (
          <div style={{
            position: "absolute", top: 10,
            left:  onLeft ? 65     : "auto",
            right: onLeft ? "auto" : 10,
            background: "rgba(10,10,15,0.95)", border: "1px solid rgba(255,85,0,0.2)",
            borderRadius: 10, padding: "10px 14px", fontSize: 12,
            pointerEvents: "none", zIndex: 10,
          }}>
            <p style={{ color: "#6b7280", marginBottom: 6 }}>{d.date}</p>
            <p style={{ color: "#9ca3af" }}>O: <span style={{ color: "#fff",                         fontWeight: 700 }}>${d.open}</span></p>
            <p style={{ color: "#9ca3af" }}>H: <span style={{ color: "#22c55e",                     fontWeight: 700 }}>${d.high}</span></p>
            <p style={{ color: "#9ca3af" }}>L: <span style={{ color: "#ef4444",                     fontWeight: 700 }}>${d.low}</span></p>
            <p style={{ color: "#9ca3af" }}>C: <span style={{ color: green ? "#22c55e" : "#ef4444", fontWeight: 700 }}>${d.close}</span></p>
          </div>
        );
      })()}
    </div>
  );
}

export default function PriceChart({ ticker }) {
  const [data, setData]           = useState([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);
  const [range, setRange]         = useState("1M");
  const [chartType, setChartType] = useState("Area");
  const [latestClose, setLatest]  = useState(null);
  const [priceChange, setChange]  = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    const days = RANGES.find(r => r.label === range)?.days || 90;
    getApiData(`/stocks/${ticker}?days=${days}`)
      .then(data => {
        const rows = data.prices.map(r => ({
          date:  r.date.slice(5),
          open:  +r.open.toFixed(2),
          high:  +r.high.toFixed(2),
          low:   +r.low.toFixed(2),
          close: +r.close.toFixed(2),
        }));
        setData(rows);
        if (rows.length >= 2) {
          const last = rows[rows.length - 1].close;
          const prev = rows[rows.length - 2].close;
          setLatest(last);
          setChange(((last - prev) / prev * 100).toFixed(2));
        }
        setLoading(false);
      })
      .catch(() => { setError("Failed to load chart data"); setLoading(false); });
  }, [ticker, range]);

  const isUp = priceChange >= 0;

  return (
    <div>
      <div className="price-header">
        <div className="price-info">
          <div className="price-ticker">{ticker} — Close Price</div>
          {latestClose && <>
            <div className="price-current">${latestClose.toLocaleString()}</div>
            <div className={`price-change ${isUp ? "up" : "down"}`}>{isUp ? "▲" : "▼"} {Math.abs(priceChange)}%</div>
          </>}
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
          <div className="time-range-group">
            {RANGES.map(r => (
              <button key={r.label} className={`time-btn ${range === r.label ? "active" : ""}`}
                      onClick={() => setRange(r.label)}>{r.label}</button>
            ))}
          </div>
          <div className="time-range-group">
            {["Area", "Candle"].map(t => (
              <button key={t} className={`time-btn ${chartType === t ? "active" : ""}`}
                      onClick={() => setChartType(t)}>{t}</button>
            ))}
          </div>
        </div>
      </div>

      {loading && <p className="loading">Loading chart...</p>}
      {error   && <p className="error">{error}</p>}

      {!loading && !error && chartType === "Candle" && (
        <CandlestickChart data={data} />
      )}

      {!loading && !error && chartType !== "Candle" && (
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <defs>
              <linearGradient id="closeGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#ff5500" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#ff5500" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fill: "#374151", fontSize: 10 }}
                   interval={Math.floor(data.length / 6)} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#374151", fontSize: 10 }} domain={["auto", "auto"]}
                   width={55} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            {chartType === "Area" && (
              <Area type="monotone" dataKey="close" stroke="#ff5500"
                    fill="url(#closeGrad)" strokeWidth={2} dot={false} name="Close" />
            )}
            {chartType === "Line" && (
              <Line type="monotone" dataKey="close" stroke="#ff5500"
                    strokeWidth={2} dot={false} name="Close" />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
