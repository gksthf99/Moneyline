-- CLV (Closing Line Value) tracking table
-- Records model probability at decision time and market closing price at tip-off
-- Used to validate whether the model consistently beats the closing line

CREATE TABLE IF NOT EXISTS clv (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    game_id BIGINT REFERENCES games(id) ON DELETE CASCADE,
    trade_id BIGINT REFERENCES trades(id) ON DELETE SET NULL,
    sport TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    team_picked TEXT NOT NULL,

    -- Model probability at decision time
    model_prob NUMERIC NOT NULL,
    market_price NUMERIC NOT NULL,
    decision_time TIMESTAMPTZ NOT NULL,

    -- Closing price (Polymarket price at game tip-off)
    closing_price NUMERIC,
    closing_time TIMESTAMPTZ,

    -- CLV calculation (filled after closing price captured)
    clv_bps NUMERIC,
    clv_pct NUMERIC,

    -- Result (filled after game resolves)
    result TEXT,  -- 'win' or 'loss'

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clv_game_id ON clv(game_id);
CREATE INDEX IF NOT EXISTS idx_clv_created_at ON clv(created_at);
CREATE INDEX IF NOT EXISTS idx_clv_sport ON clv(sport);
CREATE INDEX IF NOT EXISTS idx_clv_closing_price_null ON clv(game_id) WHERE closing_price IS NULL;
