-- Migration 001: Create core tables for Polymarket Prediction System
-- Phase 0, Group A — Database

-- 1. Games table
CREATE TABLE IF NOT EXISTS games (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sport TEXT NOT NULL CHECK (sport IN ('NBA', 'NHL')),
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    game_time TIMESTAMPTZ NOT NULL,
    triage_level TEXT NOT NULL CHECK (triage_level IN ('deep', 'standard', 'skip')),
    status TEXT NOT NULL DEFAULT 'scheduled',
    data_type TEXT,
    cached_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Research table
CREATE TABLE IF NOT EXISTS research (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    game_id BIGINT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    prob_decomposition JSONB NOT NULL DEFAULT '{}',
    edge_type TEXT CHECK (edge_type IN ('A1', 'A2', 'B', 'C')),
    effective_edge NUMERIC,
    recommendation TEXT CHECK (recommendation IN ('BET', 'PASS', 'MONITOR')),
    conditional_scenarios JSONB DEFAULT '[]',
    data_freshness_status TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Trades table
CREATE TABLE IF NOT EXISTS trades (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    game_id BIGINT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL CHECK (edge_type IN ('A1', 'A2', 'B', 'C')),
    base_prob NUMERIC NOT NULL,
    situational_adj NUMERIC NOT NULL DEFAULT 0,
    info_edge NUMERIC NOT NULL DEFAULT 0,
    final_prob NUMERIC NOT NULL,
    true_implied NUMERIC NOT NULL,
    effective_edge NUMERIC NOT NULL,
    position_size NUMERIC NOT NULL,
    entry_price NUMERIC NOT NULL,
    outcome TEXT CHECK (outcome IN ('win', 'loss', 'push', 'pending')),
    pnl NUMERIC,
    brier_contribution NUMERIC,
    confidence NUMERIC,
    conditional_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Calibration table
CREATE TABLE IF NOT EXISTS calibration (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    date DATE NOT NULL,
    trade_count INTEGER NOT NULL DEFAULT 0,
    brier_score_rolling NUMERIC,
    log_loss_favorites NUMERIC,
    log_loss_underdogs NUMERIC,
    attribution_breakdown JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    game_id BIGINT REFERENCES games(id) ON DELETE SET NULL,
    alert_type TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    delivery_channel TEXT NOT NULL,
    delivered BOOLEAN NOT NULL DEFAULT FALSE,
    fallback_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. Performance table
CREATE TABLE IF NOT EXISTS performance (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    date DATE NOT NULL,
    sport TEXT NOT NULL CHECK (sport IN ('NBA', 'NHL')),
    total_trades INTEGER NOT NULL DEFAULT 0,
    roi NUMERIC,
    pnl NUMERIC,
    brier_score NUMERIC,
    edge_type_breakdown JSONB DEFAULT '{}',
    drawdown_pct NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_games_sport_time ON games(sport, game_time);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);
CREATE INDEX IF NOT EXISTS idx_research_game_id ON research(game_id);
CREATE INDEX IF NOT EXISTS idx_trades_game_id ON trades(game_id);
CREATE INDEX IF NOT EXISTS idx_trades_edge_type ON trades(edge_type);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_game_id ON alerts(game_id);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_performance_date ON performance(date);
CREATE INDEX IF NOT EXISTS idx_calibration_date ON calibration(date);
