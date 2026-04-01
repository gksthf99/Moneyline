-- Ensemble model training data
-- Stores features + outcome for each game to enable online learning
-- Auto-retrains every 25 new rows, keeps max 150 for freshness

CREATE TABLE IF NOT EXISTS ensemble_training (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    net_rating_spread NUMERIC,
    player_impact_adj NUMERIC,
    rest_differential NUMERIC,
    home_court NUMERIC DEFAULT 1,
    pace_differential NUMERIC,
    market_price NUMERIC,
    clv_residual NUMERIC,
    home_won BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ensemble_created ON ensemble_training(created_at DESC);
