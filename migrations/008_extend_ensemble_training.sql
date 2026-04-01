-- Extend training examples with production prediction context for calibration

ALTER TABLE ensemble_training ADD COLUMN IF NOT EXISTS sport TEXT;
ALTER TABLE ensemble_training ADD COLUMN IF NOT EXISTS model_variant TEXT DEFAULT 'base';
ALTER TABLE ensemble_training ADD COLUMN IF NOT EXISTS base_probability NUMERIC;
ALTER TABLE ensemble_training ADD COLUMN IF NOT EXISTS final_probability NUMERIC;
ALTER TABLE ensemble_training ADD COLUMN IF NOT EXISTS goalie_adjustment NUMERIC;

CREATE INDEX IF NOT EXISTS idx_ensemble_sport_created ON ensemble_training(sport, created_at DESC);
