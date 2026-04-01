from datetime import date, datetime, timezone

from src.time_utils import current_sports_date, format_game_time, sports_day_bounds, CENTRAL_TZ, EASTERN_TZ


def test_sports_day_bounds_respect_dst_start():
    lower, upper = sports_day_bounds(date(2026, 3, 29))
    assert lower == "2026-03-29T10:00:00Z"
    assert upper == "2026-03-30T10:00:00Z"


def test_sports_day_bounds_respect_standard_time():
    lower, upper = sports_day_bounds(date(2026, 1, 15))
    assert lower == "2026-01-15T11:00:00Z"
    assert upper == "2026-01-16T11:00:00Z"


def test_current_sports_date_rolls_before_boundary():
    now = datetime(2026, 3, 29, 9, 30, tzinfo=timezone.utc)
    assert current_sports_date(now) == date(2026, 3, 28)


def test_format_game_time_uses_timezone_rules():
    iso = "2026-01-15T01:30:00Z"
    assert format_game_time(iso, EASTERN_TZ, "ET") == "8:30 PM ET"
    assert format_game_time(iso, CENTRAL_TZ, "CT") == "7:30 PM CT"
