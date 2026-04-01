from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

SPORTS_DAY_TZ = ZoneInfo("America/New_York")
CENTRAL_TZ = ZoneInfo("America/Chicago")
EASTERN_TZ = ZoneInfo("America/New_York")
SPORTS_DAY_BOUNDARY_HOUR = 6


def parse_utc_datetime(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def current_sports_date(now: datetime | None = None) -> date:
    """Return the current sports-day date using a 6 AM Eastern boundary."""
    current = now.astimezone(SPORTS_DAY_TZ) if now else datetime.now(SPORTS_DAY_TZ)
    if current.hour < SPORTS_DAY_BOUNDARY_HOUR:
        return (current - timedelta(days=1)).date()
    return current.date()


def sports_day_bounds(target_date: date | None = None) -> tuple[str, str]:
    """Return UTC ISO bounds for a sports day using a 6 AM Eastern boundary."""
    sports_date = target_date or current_sports_date()
    start_local = datetime.combine(
        sports_date,
        time(hour=SPORTS_DAY_BOUNDARY_HOUR),
        tzinfo=SPORTS_DAY_TZ,
    )
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc = end_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return start_utc, end_utc


def format_game_time(iso_str: str, tz: ZoneInfo, label: str) -> str:
    """Convert a UTC ISO game time into a labeled local display time."""
    if not iso_str:
        return "TBD"
    try:
        dt = parse_utc_datetime(iso_str).astimezone(tz)
        return f"{dt.strftime('%-I:%M %p')} {label}"
    except (ValueError, TypeError):
        return iso_str
