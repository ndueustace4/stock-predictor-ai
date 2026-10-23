-- database/schema.sql
-- Run once to set up all tables.
-- Safe to re-run — all statements use IF NOT EXISTS.

-- ─────────────────────────────────────────────
-- Table 1: stock_prices
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stock_prices (
    id         SERIAL PRIMARY KEY,
    ticker     VARCHAR(10)    NOT NULL,
    date       DATE           NOT NULL,
    open       NUMERIC(12, 4),
    high       NUMERIC(12, 4),
    low        NUMERIC(12, 4),
    close      NUMERIC(12, 4),
    volume     BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker ON stock_prices(ticker);
CREATE INDEX IF NOT EXISTS idx_stock_prices_date   ON stock_prices(date);


-- ─────────────────────────────────────────────
-- Table 2: news_sentiment
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS news_sentiment (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(10)    NOT NULL,
    date            DATE           NOT NULL,
    headline        TEXT,
    sentiment_score NUMERIC(5, 4),
    sentiment_label VARCHAR(10),
    source          VARCHAR(100),
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, date, headline)
);
CREATE INDEX IF NOT EXISTS idx_sentiment_ticker ON news_sentiment(ticker);
CREATE INDEX IF NOT EXISTS idx_sentiment_date   ON news_sentiment(date);
CREATE INDEX IF NOT EXISTS idx_sentiment_source ON news_sentiment(source);


-- ─────────────────────────────────────────────
-- Table 3: features  (v2 — new columns added)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS features (
    id                 SERIAL PRIMARY KEY,
    ticker             VARCHAR(10)    NOT NULL,
    date               DATE           NOT NULL,
    close              NUMERIC(12, 4),
    volume             BIGINT,

    -- Moving averages
    ma_5               NUMERIC(12, 4),
    ma_10              NUMERIC(12, 4),
    ma_20              NUMERIC(12, 4),

    -- Original features
    volatility_7       NUMERIC(12, 6),
    momentum_5         NUMERIC(12, 4),
    price_lag1         NUMERIC(12, 4),
    price_lag2         NUMERIC(12, 4),

    -- NEW: technical indicators
    rsi_14             NUMERIC(8, 4),    -- 0-100 oscillator
    macd               NUMERIC(12, 4),   -- MACD line
    macd_signal        NUMERIC(12, 4),   -- MACD signal line
    bollinger_width    NUMERIC(10, 6),   -- band width ratio
    volume_change      NUMERIC(10, 4),   -- % change in volume
    price_ma_ratio     NUMERIC(10, 6),   -- close / ma_20

    -- Sentiment (original)
    sentiment_lag1     NUMERIC(5, 4),

    -- NEW: extended sentiment features
    sentiment_lag2     NUMERIC(5, 4),    -- 2 days ago
    sentiment_ma_7     NUMERIC(5, 4),    -- 7-day rolling mean
    sentiment_std_7    NUMERIC(5, 4),    -- 7-day rolling std
    sentiment_momentum NUMERIC(5, 4),    -- lag1 - lag2

    target             SMALLINT,
    created_at         TIMESTAMP DEFAULT NOW(),

    UNIQUE (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_features_ticker ON features(ticker);
CREATE INDEX IF NOT EXISTS idx_features_date   ON features(date);


-- ─────────────────────────────────────────────
-- Add new columns to existing features table
-- (run this if you already have a features table)
-- ─────────────────────────────────────────────
ALTER TABLE features ADD COLUMN IF NOT EXISTS rsi_14             NUMERIC(8, 4);
ALTER TABLE features ADD COLUMN IF NOT EXISTS macd               NUMERIC(12, 4);
ALTER TABLE features ADD COLUMN IF NOT EXISTS macd_signal        NUMERIC(12, 4);
ALTER TABLE features ADD COLUMN IF NOT EXISTS bollinger_width    NUMERIC(10, 6);
ALTER TABLE features ADD COLUMN IF NOT EXISTS volume_change      NUMERIC(10, 4);
ALTER TABLE features ADD COLUMN IF NOT EXISTS price_ma_ratio     NUMERIC(10, 6);
ALTER TABLE features ADD COLUMN IF NOT EXISTS sentiment_lag2     NUMERIC(5, 4);
ALTER TABLE features ADD COLUMN IF NOT EXISTS sentiment_ma_7     NUMERIC(5, 4);
ALTER TABLE features ADD COLUMN IF NOT EXISTS sentiment_std_7    NUMERIC(5, 4);
ALTER TABLE features ADD COLUMN IF NOT EXISTS sentiment_momentum NUMERIC(5, 4);


-- ─────────────────────────────────────────────
-- Table 4: pipeline_monitor
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_monitor (
    id                 SERIAL PRIMARY KEY,
    run_at             TIMESTAMP DEFAULT NOW(),
    source             VARCHAR(30)  NOT NULL,
    articles_collected INT          DEFAULT 0,
    requests_made      INT          DEFAULT 0,
    windows_processed  INT          DEFAULT 0,
    notes              TEXT
);
CREATE INDEX IF NOT EXISTS idx_monitor_source ON pipeline_monitor(source);
CREATE INDEX IF NOT EXISTS idx_monitor_run_at ON pipeline_monitor(run_at);