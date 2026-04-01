-- Migration 002: Create backtest_results table for historical calibration
-- Isolated from live tables — no foreign keys to games/trades/calibration

CREATE TABLE IF NOT EXISTS backtest_results (
    id                  SERIAL PRIMARY KEY,
    sport               TEXT NOT NULL CHECK (sport IN ('nba', 'nhl')),
    game_date           DATE NOT NULL,
    home_team           TEXT NOT NULL,
    away_team           TEXT NOT NULL,
    home_wins           INTEGER NOT NULL DEFAULT 0,
    home_losses         INTEGER NOT NULL DEFAULT 0,
    away_wins           INTEGER NOT NULL DEFAULT 0,
    away_losses         INTEGER NOT NULL DEFAULT 0,
    home_win_rate       FLOAT,
    away_win_rate       FLOAT,
    home_b2b            BOOLEAN NOT NULL DEFAULT FALSE,
    away_b2b            BOOLEAN NOT NULL DEFAULT FALSE,
    home_rest_days      INTEGER,
    away_rest_days      INTEGER,
    base_prob_home      FLOAT NOT NULL,
    situational_adj     FLOAT NOT NULL,
    final_prob_home     FLOAT NOT NULL,
    triage_tier         TEXT CHECK (triage_tier IN ('skip', 'standard', 'deep')),
    home_score          INTEGER,
    away_score          INTEGER,
    actual_home_win     BOOLEAN,
    brier_score         FLOAT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_br_sport_date ON backtest_results (sport, game_date);
CREATE INDEX IF NOT EXISTS idx_br_date       ON backtest_results (game_date);
CREATE INDEX IF NOT EXISTS idx_br_triage     ON backtest_results (triage_tier);
