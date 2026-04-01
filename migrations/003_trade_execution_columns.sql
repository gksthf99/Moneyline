-- Migration 003: Add Polymarket execution columns to trades table

-- Order tracking
ALTER TABLE trades ADD COLUMN IF NOT EXISTS polymarket_order_id TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS token_id TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS team_picked TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS side TEXT DEFAULT 'BUY';
ALTER TABLE trades ADD COLUMN IF NOT EXISTS amount_usdc NUMERIC;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS shares_filled NUMERIC;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS fill_price NUMERIC;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS market_id TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS condition_id TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS sport TEXT;

-- Settlement
ALTER TABLE trades ADD COLUMN IF NOT EXISTS settled_at TIMESTAMPTZ;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS settlement_price NUMERIC;

-- Index for order lookups
CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades(polymarket_order_id);
CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome);
