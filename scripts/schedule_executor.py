#!/usr/bin/env python3
"""
Daily executor scheduler.

Runs once after the research agent. Reads today's game times from Supabase,
then schedules executor runs 30 min before each tip-off using `at`.

Usage:
    python3 scripts/schedule_executor.py          # schedule today's runs
    python3 scripts/schedule_executor.py --dry-run # show what would be scheduled
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
EXECUTOR_CMD = "/opt/sports_polymarket/run_executor.sh >> /opt/sports_polymarket/logs/executor.log 2>&1"
LEAD_TIME_MINUTES = 30


def get_todays_game_times() -> list[dict]:
    """Fetch today's game times from Supabase."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

    now = datetime.now(timezone.utc)
    # Games from now through end of tomorrow (catches late games past midnight)
    start = now.isoformat()
    end = (now + timedelta(hours=24)).isoformat()

    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/games",
        headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
        params={
            "select": "id,sport,home_team,away_team,game_time",
            "game_time": f"gte.{start}",
            "order": "game_time.asc",
            "limit": "50",
        },
        timeout=15,
    )
    resp.raise_for_status()
    games = resp.json()

    # Filter to games within 24h
    result = []
    for g in games:
        gt = g.get("game_time", "")
        if not gt:
            continue
        try:
            game_dt = datetime.fromisoformat(gt.replace("Z", "+00:00"))
            if game_dt <= now + timedelta(hours=24):
                g["game_dt"] = game_dt
                result.append(g)
        except (ValueError, TypeError):
            continue

    return result


def clear_pending_at_jobs():
    """Remove any previously scheduled executor at jobs."""
    try:
        result = subprocess.run(["atq"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            job_id = line.split()[0]
            # Check if this job is our executor
            check = subprocess.run(
                ["at", "-c", job_id], capture_output=True, text=True, timeout=5
            )
            if "run_executor.sh" in check.stdout:
                subprocess.run(["atrm", job_id], capture_output=True, timeout=5)
                logger.info("Removed old at job %s", job_id)
    except Exception as e:
        logger.warning("Could not clear old at jobs: %s", e)


def schedule_run(run_time: datetime, game_label: str, dry_run: bool = False) -> bool:
    """Schedule an executor run at a specific time using `at`."""
    time_str = run_time.strftime("%H:%M %Y-%m-%d")

    if dry_run:
        logger.info("  [DRY RUN] Would schedule at %s UTC for %s", time_str, game_label)
        return True

    try:
        proc = subprocess.run(
            ["at", "-M", time_str],
            input=EXECUTOR_CMD,
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "TZ": "UTC"},
        )
        if proc.returncode == 0:
            logger.info("  Scheduled at %s UTC for %s", time_str, game_label)
            return True
        else:
            logger.warning("  at failed: %s", proc.stderr.strip())
            return False
    except Exception as e:
        logger.warning("  Failed to schedule: %s", e)
        return False


def main():
    dry_run = "--dry-run" in sys.argv

    games = get_todays_game_times()
    if not games:
        logger.info("No upcoming games found")
        return

    logger.info("Found %d games today", len(games))

    if not dry_run:
        clear_pending_at_jobs()

    now = datetime.now(timezone.utc)
    scheduled = 0

    # Group games by run time (dedupe to avoid multiple runs at the same minute)
    run_times = {}
    for g in games:
        game_dt = g["game_dt"]
        run_dt = game_dt - timedelta(minutes=LEAD_TIME_MINUTES)

        if run_dt <= now:
            logger.info("  Skipping %s @ %s — already past 30min window",
                        g["away_team"], g["home_team"])
            continue

        # Round to nearest 1 min
        run_key = run_dt.strftime("%Y-%m-%d %H:%M")
        label = f"{g['sport']} {g['away_team']} @ {g['home_team']} ({game_dt.strftime('%H:%M')} UTC)"

        if run_key not in run_times:
            run_times[run_key] = {"dt": run_dt, "games": []}
        run_times[run_key]["games"].append(label)

    for run_key, info in sorted(run_times.items()):
        labels = ", ".join(info["games"])
        if schedule_run(info["dt"], labels, dry_run=dry_run):
            scheduled += 1

    logger.info("Scheduled %d executor runs for %d games", scheduled, len(games))


if __name__ == "__main__":
    main()
