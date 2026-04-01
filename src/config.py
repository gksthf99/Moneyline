import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# API Keys
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TENNIS_DATA_API_KEY = os.getenv("TENNIS_DATA_API_KEY", "")
TENNIS_DATA_BASE_URL = os.getenv("TENNIS_DATA_BASE_URL", "")
TENNIS_ODDS_BASE_URL = os.getenv("TENNIS_ODDS_BASE_URL", "")
TENNIS_PROVIDER = os.getenv("TENNIS_PROVIDER", "sportradar")
TENNIS_SPORTRADAR_ACCESS_LEVEL = os.getenv("TENNIS_SPORTRADAR_ACCESS_LEVEL", "trial")
TENNIS_SPORTRADAR_LANGUAGE = os.getenv("TENNIS_SPORTRADAR_LANGUAGE", "en")
TENNIS_ODDS_API_KEY = os.getenv("TENNIS_ODDS_API_KEY", "")

# Cache TTLs (seconds)
CACHE_TTL = {
    "team_season_stats": 86400,    # 24 hours
    "injury_reports": 1800,         # 30 minutes
    "lineup_goalie": 0,             # never cache — always live
    "historical_h2h": 604800,       # 7 days
    "market_odds": 300,             # 5 minutes
    "schedule": 3600,               # 1 hour
    "player_stats": 3600,           # 1 hour
    "standings": 3600,              # 1 hour
    "team_advanced": 3600,          # 1 hour (Corsi, xG)
}
