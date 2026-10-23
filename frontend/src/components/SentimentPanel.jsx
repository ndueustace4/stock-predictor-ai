import { useEffect, useState } from "react";
import { getApiData } from "../api";

export default function SentimentPanel({ ticker }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    getApiData(`/news/${ticker}?days=15`)
      .then(result => { setData(result); setLoading(false); })
      .catch(() => { setError("Failed to load news"); setLoading(false); });
  }, [ticker]);

  const avg   = data?.demo_snapshot
    ? data.average_sentiment
    : data?.news?.length
      ? data.news.reduce((a,b)=>a+b.sentiment_score,0)/data.news.length
      : 0;
  const label = data?.demo_snapshot
    ? data.sentiment_label
    : avg >= 0.05 ? "positive" : avg <= -0.05 ? "negative" : "neutral";

  return (
    <div className="card">
      <h2>News Sentiment — {ticker}</h2>
      {loading && <p className="loading">Loading news...</p>}
      {error   && <p className="error">{error}</p>}
      {!loading && !error && data && (
        <>
        <div className="sentiment-score">
          <div className={`sentiment-number ${label}`}>{avg>=0?"+":""}{avg.toFixed(3)}</div>
          <div className="sentiment-label-text">{label} sentiment</div>
          <div className="sentiment-count">{data.count} recent headlines</div>
        </div>
        {data.demo_snapshot && (
          <div className="news-list">
            <p className="loading">
              Headline text is omitted from this public preview.
              {data.sentiment_days?.[0] && <> Latest saved sentiment: {data.sentiment_days[0].date}.</>}
            </p>
            {data.sentiment_days?.map(day => (
              <div className="news-item" key={day.date}>
                <div className="news-meta">
                  {day.date} &nbsp;·&nbsp; {day.count} saved {day.count === 1 ? "score" : "scores"}
                </div>
                <div className="news-headline">
                  Average sentiment: {day.average_sentiment >= 0 ? "+" : ""}{day.average_sentiment.toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        )}
        {!data.demo_snapshot && (
          <div className="news-list">
            {data.news.slice(0,8).map((item,i) => (
              <div className={`news-item ${item.sentiment_label}`} key={i}>
                <div className="news-headline">{item.headline}</div>
                <div className="news-meta">
                  {item.date} &nbsp;·&nbsp;
                  <span style={{color:item.sentiment_label==="positive"?"#22c55e":item.sentiment_label==="negative"?"#ef4444":"#6b7280"}}>
                    {item.sentiment_label} ({item.sentiment_score>=0?"+":""}{item.sentiment_score.toFixed(3)})
                  </span>
                </div>
              </div>
            ))}
            {data.count === 0 && <p className="loading">No recent news for {ticker}</p>}
          </div>
        )}
        </>
      )}
    </div>
  );
}
