// frontend/src/components/StockSelector.jsx
export default function StockSelector({ tickers, selected, onChange }) {
  return (
    <select
      className="stock-select"
      value={selected}
      onChange={e => onChange(e.target.value)}
    >
      {tickers.map(t => (
        <option key={t} value={t}>{t}</option>
      ))}
    </select>
  );
}
