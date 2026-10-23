import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL?.replace(/\/$/, "");
export const isStaticPreview = !API_URL;

let snapshotPromise;

export function loadDemoSnapshot() {
  if (!snapshotPromise) {
    snapshotPromise = fetch(`${import.meta.env.BASE_URL}demo-data.json`).then(response => {
      if (!response.ok) {
        throw new Error(`Could not load portfolio snapshot (${response.status})`);
      }
      return response.json();
    });
  }
  return snapshotPromise;
}

export async function getApiData(path) {
  if (API_URL) {
    const response = await axios.get(`${API_URL}${path}`);
    return response.data;
  }

  const url = new URL(path, "http://portfolio-preview.local");
  const [, route, ticker] = url.pathname.split("/");
  const snapshot = await loadDemoSnapshot();
  const stock = snapshot.tickers[ticker];

  if (!stock) {
    throw new Error(`No portfolio snapshot is available for ${ticker}`);
  }

  if (route === "stocks") {
    const days = Number(url.searchParams.get("days") || 90);
    const prices = stock.prices.slice(-days);
    return { ticker, days, count: prices.length, prices };
  }

  if (route === "news") {
    const average = stock.sentiment.average_sentiment;
    return {
      ticker,
      count: stock.sentiment.count,
      news: [],
      demo_snapshot: true,
      sentiment_label: average >= 0.05 ? "positive" : average <= -0.05 ? "negative" : "neutral",
      ...stock.sentiment,
    };
  }

  if (route === "predict") {
    const horizon = url.searchParams.get("horizon") || "1d";
    const useSentiment = url.searchParams.get("use_sentiment") !== "false";
    return stock.predictions[horizon]?.[useSentiment ? "with_sentiment" : "no_sentiment"];
  }

  if (route === "compare") {
    const horizon = url.searchParams.get("horizon") || "1d";
    const noSentiment = stock.predictions[horizon]?.no_sentiment;
    const withSentiment = stock.predictions[horizon]?.with_sentiment;
    if (!noSentiment || !withSentiment) {
      throw new Error(`No ${horizon} comparison is available for ${ticker}`);
    }
    return {
      ticker,
      horizon,
      model_without_sentiment: {
        prediction: noSentiment.prediction,
        confidence: noSentiment.confidence,
        top_factors: noSentiment.factors.slice(0, 3),
        plain_english: noSentiment.plain_english,
      },
      model_with_sentiment: {
        prediction: withSentiment.prediction,
        confidence: withSentiment.confidence,
        top_factors: withSentiment.factors.slice(0, 3),
        plain_english: withSentiment.plain_english,
      },
    };
  }

  if (route === "analysis") {
    const horizon = url.searchParams.get("horizon") || "1d";
    const prediction = stock.predictions[horizon]?.with_sentiment;
    if (!prediction) {
      throw new Error(`No ${horizon} prediction is available for ${ticker}`);
    }
    const driver = prediction.factors[0];
    return {
      ticker,
      prediction: prediction.prediction,
      confidence: Number((prediction.confidence * 100).toFixed(1)),
      analysis: {
        main_reason: driver
          ? `${driver.label} ${driver.shap_value >= 0 ? "supported" : "weighed against"} the model's ${prediction.prediction} prediction.`
          : "The model evaluated the latest saved market features.",
        news_impact: `The saved sentiment score is ${stock.sentiment.average_sentiment >= 0 ? "positive" : "negative"} (${stock.sentiment.average_sentiment.toFixed(3)}). Individual headlines are not included in this public snapshot.`,
        confidence_reason: `The model assigned ${(prediction.confidence * 100).toFixed(1)}% probability to its predicted class. This is model confidence, not a guarantee.`,
      },
      demo_snapshot: true,
    };
  }

  throw new Error(`Unsupported static preview endpoint: ${route}`);
}
