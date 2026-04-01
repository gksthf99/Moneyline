-- Immutable point-in-time prediction lineage

CREATE TABLE IF NOT EXISTS prediction_snapshots (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    game_id BIGINT REFERENCES games(id) ON DELETE SET NULL,
    sport TEXT NOT NULL CHECK (sport IN ('NBA', 'NHL')),
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    game_time TIMESTAMPTZ,
    triage_level TEXT,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    snapshot JSONB NOT NULL DEFAULT '{}',
    feature_snapshot JSONB NOT NULL DEFAULT '{}',
    model_variant TEXT NOT NULL DEFAULT 'base',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS prediction_records (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    game_id BIGINT REFERENCES games(id) ON DELETE SET NULL,
    sport TEXT NOT NULL CHECK (sport IN ('NBA', 'NHL')),
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    game_time TIMESTAMPTZ,
    triage_level TEXT,
    base_probability NUMERIC NOT NULL,
    situational_adjustment NUMERIC NOT NULL DEFAULT 0,
    information_edge NUMERIC NOT NULL DEFAULT 0,
    final_probability NUMERIC NOT NULL,
    edge_type TEXT CHECK (edge_type IN ('A1', 'A2', 'B', 'C')),
    recommendation TEXT CHECK (recommendation IN ('BET', 'PASS', 'MONITOR')),
    effective_edge NUMERIC,
    bet_side TEXT,
    model_family TEXT NOT NULL DEFAULT 'layered_rule_model',
    model_variant TEXT NOT NULL DEFAULT 'base',
    decomposition JSONB NOT NULL DEFAULT '{}',
    feature_snapshot JSONB NOT NULL DEFAULT '{}',
    snapshot_metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prediction_snapshots_game_id ON prediction_snapshots(game_id);
CREATE INDEX IF NOT EXISTS idx_prediction_snapshots_created_at ON prediction_snapshots(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_records_game_id ON prediction_records(game_id);
CREATE INDEX IF NOT EXISTS idx_prediction_records_created_at ON prediction_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_records_model_variant ON prediction_records(model_variant);
