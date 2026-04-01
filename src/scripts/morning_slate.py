"""
Morning Slate Script — posts daily schedule to #daily-schedule.

Deterministic Python. NO LLM. Cron-scheduled at 7:00 AM daily.

Pulls:
- NBA + NHL games for today
- Rest differential per team (days since last game)
- B2B flags (played yesterday)
- Travel flags (cross-timezone within 24h)
- Goalie status where confirmed (NHL)
- Triage classification: Deep Dive / Standard / Skip

Posts to Discord via webhook.
"""

import json
import os
import sys
import requests
from datetime import date, datetime, timedelta
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.nba_espn import get_scoreboard as nba_scoreboard
from src.data.nhl_api import get_schedule as nhl_schedule, get_standings as nhl_standings
from src.data.nhl_dailyfaceoff import get_starting_goalies
from src.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

# ---------------------------------------------------------------------------
# Team timezone mappings (US timezone offset from ET)
# 0 = Eastern, -1 = Central, -2 = Mountain, -3 = Pacific
# ---------------------------------------------------------------------------

NBA_TIMEZONES = {
    # Eastern (0)
    "Boston Celtics": 0, "Brooklyn Nets": 0, "New York Knicks": 0,
    "Philadelphia 76ers": 0, "Toronto Raptors": 0, "Chicago Bulls": -1,
    "Cleveland Cavaliers": 0, "Detroit Pistons": 0, "Indiana Pacers": 0,
    "Milwaukee Bucks": -1, "Atlanta Hawks": 0, "Charlotte Hornets": 0,
    "Miami Heat": 0, "Orlando Magic": 0, "Washington Wizards": 0,
    # Central (-1)
    "Dallas Mavericks": -1, "Houston Rockets": -1, "Memphis Grizzlies": -1,
    "New Orleans Pelicans": -1, "San Antonio Spurs": -1,
    "Minnesota Timberwolves": -1, "Oklahoma City Thunder": -1,
    # Mountain (-2)
    "Denver Nuggets": -2, "Utah Jazz": -2, "Phoenix Suns": -2,
    # Pacific (-3)
    "Golden State Warriors": -3, "LA Clippers": -3,
    "Los Angeles Lakers": -3, "Sacramento Kings": -3,
    "Portland Trail Blazers": -3, "Seattle SuperSonics": -3,
}

NHL_TIMEZONES = {
    # Eastern (0)
    "Boston Bruins": 0, "Buffalo Sabres": 0, "Carolina Hurricanes": 0,
    "Columbus Blue Jackets": 0, "Detroit Red Wings": 0,
    "Florida Panthers": 0, "Montreal Canadiens": 0,
    "New Jersey Devils": 0, "New York Islanders": 0,
    "New York Rangers": 0, "Ottawa Senators": 0,
    "Philadelphia Flyers": 0, "Pittsburgh Penguins": 0,
    "Tampa Bay Lightning": 0, "Toronto Maple Leafs": 0,
    "Washington Capitals": 0,
    # Central (-1)
    "Chicago Blackhawks": -1, "Colorado Avalanche": -2,
    "Dallas Stars": -1, "Minnesota Wild": -1,
    "Nashville Predators": -1, "St. Louis Blues": -1,
    "Utah Hockey Club": -2, "Winnipeg Jets": -1,
    # Pacific (-3)
    "Anaheim Ducks": -3, "Calgary Flames": -2,
    "Edmonton Oilers": -2, "Los Angeles Kings": -3,
    "San Jose Sharks": -3, "Seattle Kraken": -3,
    "Vancouver Canucks": -3, "Vegas Golden Knights": -3,
}

# ---------------------------------------------------------------------------
# Triage classification heuristics
# ---------------------------------------------------------------------------

def _classify_game(sport: str, home: str, away: str, has_b2b: bool,
                    has_travel: bool, has_goalie_info: bool) -> str:
    """Classify a game as Deep Dive / Standard / Skip.

    Purely factor-driven — games with situational edges get prioritized.
    """
    situational_count = sum([has_b2b, has_travel])

    # Multiple situational factors → Deep Dive
    if situational_count >= 2:
        return "Deep Dive"

    # One situational factor → Deep Dive
    if situational_count == 1:
        return "Deep Dive"

    # NHL: confirmed goalie info adds analytical value
    if sport == "NHL" and has_goalie_info:
        return "Standard"

    # No situational factors
    return "Standard"


# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

def _nhl_str(val) -> str:
    """Extract string from NHL API field (may be str or {'default': str})."""
    if isinstance(val, dict):
        return val.get("default", "")
    return val or ""


def _get_nba_yesterday_teams() -> set[str]:
    """Get teams that played yesterday (for B2B detection)."""
    yesterday = date.today() - timedelta(days=1)
    try:
        games = nba_scoreboard(yesterday)
        teams = set()
        for g in games:
            if g.get("home_team"):
                teams.add(g["home_team"])
            if g.get("away_team"):
                teams.add(g["away_team"])
        return teams
    except Exception:
        return set()


def _get_nhl_yesterday_teams() -> set[str]:
    """Get NHL teams that played yesterday (for B2B detection)."""
    yesterday = date.today() - timedelta(days=1)
    try:
        data = nhl_schedule(yesterday)
        teams = set()
        for day in data.get("gameWeek", []):
            if day.get("date") == yesterday.isoformat():
                for game in day.get("games", []):
                    away_place = _nhl_str(game.get("awayTeam", {}).get("placeName"))
                    away_name = _nhl_str(game.get("awayTeam", {}).get("commonName"))
                    home_place = _nhl_str(game.get("homeTeam", {}).get("placeName"))
                    home_name = _nhl_str(game.get("homeTeam", {}).get("commonName"))
                    if away_place and away_name:
                        teams.add(f"{away_place} {away_name}")
                    if home_place and home_name:
                        teams.add(f"{home_place} {home_name}")
        return teams
    except Exception:
        return set()


def _travel_zones(team_name: str, sport: str) -> int:
    """Get timezone offset for a team. Returns 0 if unknown."""
    tz_map = NBA_TIMEZONES if sport == "NBA" else NHL_TIMEZONES
    return tz_map.get(team_name, 0)


def _compute_travel_flag(home: str, away: str, sport: str) -> tuple[bool, int]:
    """Check if away team crossed 2+ timezones. Returns (flag, zones_crossed)."""
    home_tz = _travel_zones(home, sport)
    away_tz = _travel_zones(away, sport)
    zones = abs(home_tz - away_tz)
    return zones >= 2, zones


# ---------------------------------------------------------------------------
# Slate builders
# ---------------------------------------------------------------------------

def build_nba_slate() -> list[dict]:
    """Build NBA slate for today."""
    today = date.today()
    games = nba_scoreboard(today)
    yesterday_teams = _get_nba_yesterday_teams()

    slate = []
    for g in games:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        game_time = g.get("date", "")

        home_b2b = home in yesterday_teams
        away_b2b = away in yesterday_teams
        has_b2b = home_b2b or away_b2b

        travel_flag, travel_zones = _compute_travel_flag(home, away, "NBA")

        triage = _classify_game("NBA", home, away, has_b2b, travel_flag, False)

        # Rest differential (simplified: B2B = 1 day rest, otherwise assume 2)
        home_rest = 1 if home_b2b else 2
        away_rest = 1 if away_b2b else 2

        slate.append({
            "sport": "NBA",
            "home_team": home,
            "away_team": away,
            "home_abbrev": g.get("home_team_abbrev", ""),
            "away_abbrev": g.get("away_team_abbrev", ""),
            "game_time": game_time,
            "home_b2b": home_b2b,
            "away_b2b": away_b2b,
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "travel_flag": travel_flag,
            "travel_zones": travel_zones,
            "triage": triage,
        })

    return slate


def build_nhl_slate() -> list[dict]:
    """Build NHL slate for today."""
    today = date.today()
    schedule_data = nhl_schedule(today)
    yesterday_teams = _get_nhl_yesterday_teams()

    # Get goalie starts
    try:
        goalies = get_starting_goalies(today)
    except Exception:
        goalies = []

    # Index goalies by team name (fuzzy — match on partial)
    goalie_map = {}
    for g in goalies:
        goalie_map[g.get("away_team", "")] = g.get("away_goalie")
        goalie_map[g.get("home_team", "")] = g.get("home_goalie")

    slate = []
    for day in schedule_data.get("gameWeek", []):
        if day.get("date") != today.isoformat():
            continue
        for game in day.get("games", []):
            away_place = _nhl_str(game.get("awayTeam", {}).get("placeName"))
            away_name = _nhl_str(game.get("awayTeam", {}).get("commonName"))
            home_place = _nhl_str(game.get("homeTeam", {}).get("placeName"))
            home_name = _nhl_str(game.get("homeTeam", {}).get("commonName"))

            away_full = f"{away_place} {away_name}".strip()
            home_full = f"{home_place} {home_name}".strip()

            away_abbrev = game.get("awayTeam", {}).get("abbrev", "")
            home_abbrev = game.get("homeTeam", {}).get("abbrev", "")

            start_time = game.get("startTimeUTC", "")

            home_b2b = home_full in yesterday_teams
            away_b2b = away_full in yesterday_teams
            has_b2b = home_b2b or away_b2b

            travel_flag, travel_zones = _compute_travel_flag(home_full, away_full, "NHL")

            # Goalie lookup — try full name, then partial matches
            home_goalie = goalie_map.get(home_full)
            away_goalie = goalie_map.get(away_full)
            if not home_goalie:
                for key, val in goalie_map.items():
                    if home_name and home_name in key:
                        home_goalie = val
                        break
            if not away_goalie:
                for key, val in goalie_map.items():
                    if away_name and away_name in key:
                        away_goalie = val
                        break

            has_goalie_info = bool(home_goalie or away_goalie)
            triage = _classify_game("NHL", home_full, away_full, has_b2b,
                                    travel_flag, has_goalie_info)

            home_rest = 1 if home_b2b else 2
            away_rest = 1 if away_b2b else 2

            slate.append({
                "sport": "NHL",
                "home_team": home_full,
                "away_team": away_full,
                "home_abbrev": home_abbrev,
                "away_abbrev": away_abbrev,
                "game_time": start_time,
                "home_b2b": home_b2b,
                "away_b2b": away_b2b,
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "travel_flag": travel_flag,
                "travel_zones": travel_zones,
                "home_goalie": home_goalie,
                "away_goalie": away_goalie,
                "triage": triage,
            })

    return slate


# ---------------------------------------------------------------------------
# Discord embed formatter
# ---------------------------------------------------------------------------

def _format_time(iso_str: str) -> str:
    """Convert ISO datetime to readable ET time."""
    if not iso_str:
        return "TBD"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        # Convert UTC to ET (approximate: UTC-4 for EDT, UTC-5 for EST)
        # Use -4 during NBA/NHL season (March-June is EDT)
        et = dt - timedelta(hours=4)
        return et.strftime("%-I:%M %p ET")
    except (ValueError, TypeError):
        return iso_str[:16] if len(iso_str) > 16 else iso_str


def _b2b_indicator(is_b2b: bool) -> str:
    return " B2B" if is_b2b else ""


def _triage_emoji(triage: str) -> str:
    if triage == "Deep Dive":
        return "[DEEP]"
    elif triage == "Standard":
        return "[STD]"
    else:
        return "[SKIP]"


def format_slate_embed(nba_slate: list[dict], nhl_slate: list[dict]) -> dict:
    """Format full slate as a Discord webhook embed."""
    today_str = date.today().strftime("%A, %B %-d, %Y")

    # Count by triage
    all_games = nba_slate + nhl_slate
    deep = sum(1 for g in all_games if g["triage"] == "Deep Dive")
    standard = sum(1 for g in all_games if g["triage"] == "Standard")
    skip = sum(1 for g in all_games if g["triage"] == "Skip")

    description = f"**{len(all_games)} games** | Deep: {deep} | Standard: {standard} | Skip: {skip}"

    fields = []

    def _build_sport_lines(slate: list[dict], sport: str) -> list[str]:
        """Build formatted lines for a sport's slate."""
        triage_order = {"Deep Dive": 0, "Standard": 1, "Skip": 2}
        sorted_slate = sorted(slate, key=lambda g: (triage_order.get(g["triage"], 1), g.get("game_time", "")))
        lines = []
        for g in sorted_slate:
            time_str = _format_time(g["game_time"])
            away_b2b = _b2b_indicator(g["away_b2b"])
            home_b2b = _b2b_indicator(g["home_b2b"])
            triage = _triage_emoji(g["triage"])
            line = f"{triage} **{g['away_abbrev']}{away_b2b}** @ **{g['home_abbrev']}{home_b2b}** — {time_str}"
            extras = []
            if g["away_b2b"] or g["home_b2b"]:
                extras.append(f"Rest: {g['away_abbrev']} {g['away_rest_days']}d / {g['home_abbrev']} {g['home_rest_days']}d")
            if g["travel_flag"]:
                extras.append(f"Travel: {g['away_abbrev']} crossed {g['travel_zones']} zones")
            if sport == "NHL":
                goalie_parts = []
                if g.get("away_goalie"):
                    goalie_parts.append(f"{g['away_abbrev']}: {g['away_goalie']}")
                if g.get("home_goalie"):
                    goalie_parts.append(f"{g['home_abbrev']}: {g['home_goalie']}")
                if goalie_parts:
                    extras.append("G: " + " / ".join(goalie_parts))
            if extras:
                line += "\n  " + " | ".join(extras)
            lines.append(line)
        return lines

    def _add_fields_split(fields_list: list, label: str, lines: list[str]) -> None:
        """Add field(s) to embed, splitting if value exceeds 1024 chars."""
        if not lines:
            return
        chunks = []
        current = []
        current_len = 0
        for line in lines:
            added_len = len(line) + (1 if current else 0)  # +1 for newline
            if current_len + added_len > 1000 and current:
                chunks.append("\n".join(current))
                current = [line]
                current_len = len(line)
            else:
                current.append(line)
                current_len += added_len
        if current:
            chunks.append("\n".join(current))
        for i, chunk in enumerate(chunks):
            name = label if i == 0 else f"{label} (cont.)"
            fields_list.append({"name": name, "value": chunk, "inline": False})

    # NBA section
    if nba_slate:
        nba_lines = _build_sport_lines(nba_slate, "NBA")
        _add_fields_split(fields, f"NBA ({len(nba_slate)} games)", nba_lines)

    # NHL section
    if nhl_slate:
        nhl_lines = _build_sport_lines(nhl_slate, "NHL")
        _add_fields_split(fields, f"NHL ({len(nhl_slate)} games)", nhl_lines)

    if not nba_slate and not nhl_slate:
        fields.append({
            "name": "No Games Today",
            "value": "No NBA or NHL games scheduled.",
            "inline": False,
        })

    embed = {
        "title": f"Morning Slate — {today_str}",
        "description": description,
        "color": 0x5865F2,  # Discord blurple
        "fields": fields,
        "footer": {"text": "Triage: [DEEP] = Deep Dive | [STD] = Standard | [SKIP] = Skip (calibration only)"},
        "timestamp": datetime.now(tz=None).astimezone().isoformat(),
    }

    return embed


# ---------------------------------------------------------------------------
# Supabase: insert games into games table
# ---------------------------------------------------------------------------

def _insert_games_to_supabase(slate: list[dict]) -> None:
    """Insert today's slate into the games table for downstream agents."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("[WARN] Supabase not configured — skipping game inserts.")
        return

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        # Upsert: on conflict with uq_games_matchup, ignore duplicate
        "Prefer": "resolution=ignore-duplicates",
    }

    rows = []
    for g in slate:
        # DB constraint: sport IN ('NBA','NHL'), triage IN ('deep','standard','skip')
        triage_db = g["triage"].lower().split()[0]  # "Deep Dive" -> "deep"
        rows.append({
            "sport": g["sport"].upper(),
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "game_time": g["game_time"],
            "triage_level": triage_db,
            "status": "scheduled",
        })

    if not rows:
        return

    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/games",
            headers=headers,
            json=rows,
            timeout=30,
        )
        if resp.status_code in (200, 201):
            print(f"[OK] Inserted/skipped {len(rows)} games to Supabase")
        else:
            print(f"[WARN] Supabase batch insert: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[WARN] Supabase insert error: {e}")


# ---------------------------------------------------------------------------
# Discord webhook posting
# ---------------------------------------------------------------------------

def post_to_discord(embed: dict, webhook_url: str) -> bool:
    """Post embed to Discord via webhook. Returns True on success."""
    payload = {
        "username": "Morning Slate",
        "embeds": [embed],
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=15)
        if resp.status_code in (200, 204):
            print("[OK] Slate posted to #daily-schedule")
            return True
        else:
            print(f"[ERROR] Discord webhook failed: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Discord webhook error: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Build and post the morning slate."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building morning slate for {date.today().isoformat()}...")

    # Build slates
    nba_slate = build_nba_slate()
    print(f"  NBA: {len(nba_slate)} games")

    nhl_slate = build_nhl_slate()
    print(f"  NHL: {len(nhl_slate)} games")

    all_games = nba_slate + nhl_slate
    deep = sum(1 for g in all_games if g["triage"] == "Deep Dive")
    standard = sum(1 for g in all_games if g["triage"] == "Standard")
    skip = sum(1 for g in all_games if g["triage"] == "Skip")
    print(f"  Triage: {deep} deep, {standard} standard, {skip} skip")

    # Insert games to Supabase
    _insert_games_to_supabase(all_games)

    # Format and post to Discord
    embed = format_slate_embed(nba_slate, nhl_slate)

    # Load webhook URL
    config_path = Path(__file__).resolve().parent.parent.parent / ".discord_config.json"
    if not config_path.exists():
        print(f"[ERROR] Discord config not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        discord_config = json.load(f)

    webhook_url = discord_config.get("webhooks", {}).get("daily_schedule")
    if not webhook_url:
        print("[ERROR] No daily_schedule webhook URL in .discord_config.json")
        sys.exit(1)

    success = post_to_discord(embed, webhook_url)

    if not success:
        print("[FALLBACK] Printing slate to stdout:")
        print(json.dumps(embed, indent=2))
        sys.exit(1)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()
