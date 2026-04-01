-- Add discord_thread_id to research table for alert agent thread replies
ALTER TABLE research ADD COLUMN IF NOT EXISTS discord_thread_id TEXT;
