"""
Performance Agent (Jerry) — basic skeleton.

On game resolution:
  1. Pull stored predictions from Supabase research table
  2. Pull final scores from ESPN (NBA) / NHL API
  3. Compare predicted probability to actual outcome
  4. Calculate Brier score per game
  5. Write to Supabase calibration table
  6. Post summary to #performance

Every resolved game without a grade is calibration data lost forever.
"""

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from src.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from src.data.nba_espn import get_scoreboard as nba_scoreboard
from src.data.nhl_api import get_daily_scores as nhl_scores
from src.domain.prediction_record import GradingRecord, PredictionRecord
from src.repositories.ensemble_repository import EnsembleTrainingRepository
from src.services.training_dataset import build_ensemble_training_row
from src.time_utils import sports_day_bounds

logger = logging.getLogger(__name__)
_ensemble_repo = EnsembleTrainingRepository()


# ---------------------------------------------------------------------------
# Fetch resolved games
# ---------------------------------------------------------------------------

def get_resolved_nba(game_date: date) -> list[dict]:
    """Get final NBA scores for a date."""
    games = nba_scoreboard(game_date)
    resolved = []
    for g in games:
        if g.get("status") == "Final":
            home_score = int(g.get("home_score", 0) or 0)
            away_score = int(g.get("away_score", 0) or 0)
            resolved.append({
                "home_team": g.get("home_team", ""),
                "away_team": g.get("away_team", ""),
                "home_score": home_score,
                "away_score": away_score,
                "home_won": home_score > away_score,
            })
    return resolved


_NHL_ABBREV_TO_FULL = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames", "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche", "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers", "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montréal Canadiens",
    "NSH": "Nashville Predators", "NJD": "New Jersey Devils", "NYI": "New York Islanders",
    "NYR": "New York Rangers", "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks", "SEA": "Seattle Kraken",
    "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}


def get_resolved_nhl(game_date: date) -> list[dict]:
    """Get final NHL scores for a date."""
    try:
        data = nhl_scores(game_date)
    except Exception as e:
        logger.warning("Failed to fetch NHL scores: %s", e)
        return []

    resolved = []
    for game in data.get("games", []):
        state = game.get("gameState", "")
        if state not in ("FINAL", "OFF"):
            continue

        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})
        home_score = home.get("score", 0)
        away_score = away.get("score", 0)

        home_name = _NHL_ABBREV_TO_FULL.get(home.get("abbrev", ""), "")
        away_name = _NHL_ABBREV_TO_FULL.get(away.get("abbrev", ""), "")

        # Fallback: use name field from API
        if not home_name:
            n = home.get("name", {})
            home_name = n.get("default", str(n)) if isinstance(n, dict) else str(n)
        if not away_name:
            n = away.get("name", {})
            away_name = n.get("default", str(n)) if isinstance(n, dict) else str(n)

        resolved.append({
            "home_team": home_name,
            "away_team": away_name,
            "home_score": home_score,
            "away_score": away_score,
            "home_won": home_score > away_score,
        })
    return resolved


# ---------------------------------------------------------------------------
# Fetch stored predictions from Supabase
# ---------------------------------------------------------------------------

def get_predictions(game_date: date) -> list[dict]:
    """Get research predictions for games on a given date."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return []

    date_start = f"{game_date.isoformat()}T00:00:00"
    date_end = f"{(game_date + timedelta(days=1)).isoformat()}T12:00:00"

    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/research",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            },
            params={
                "select": "id,game_id,prob_decomposition,edge_type,effective_edge,recommendation,created_at",
                "created_at": f"gte.{date_start}",
                "order": "created_at.asc",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch predictions: %s", e)

    return []


def parse_prediction_rows(rows: list[dict], game_map: dict[int, dict]) -> list[PredictionRecord]:
    parsed = []
    for row in rows:
        record = PredictionRecord.from_supabase_row(row, game_map.get(row.get("game_id")))
        if record:
            parsed.append(record)
    return parsed


def persist_training_examples(pairs: list[tuple[PredictionRecord, GradingRecord]]) -> int:
    stored = 0
    for prediction, grade in pairs:
        if prediction.model_variant != "base":
            continue
        if not prediction.feature_snapshot:
            continue
        row = build_ensemble_training_row(prediction, grade)
        if _ensemble_repo.store_example(row):
            stored += 1
    return stored


def get_games_from_supabase(game_date: date) -> list[dict]:
    """Get games from Supabase for matching predictions to outcomes.

    Uses a 6 AM Eastern sports-day boundary so late games stay on the
    intended slate across DST changes.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return []

    lower, upper = sports_day_bounds(game_date)

    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/games",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            },
            params={
                "and": f"(game_time.gte.{lower},game_time.lt.{upper})",
                "select": "id,sport,home_team,away_team,game_time",
                "order": "game_time.asc",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch games: %s", e)

    return []


# ---------------------------------------------------------------------------
# Brier score calculation
# ---------------------------------------------------------------------------

def brier_score(predicted_prob: float, outcome: int) -> float:
    """Brier score for a single prediction. Lower is better.
    outcome: 1 if predicted event happened, 0 if not.
    """
    return (predicted_prob - outcome) ** 2


# ---------------------------------------------------------------------------
# Grade a single game
# ---------------------------------------------------------------------------

def grade_game(prediction: PredictionRecord, outcome: dict) -> GradingRecord | None:
    """Grade a prediction against its actual outcome.

    Returns grading dict or None if can't match.
    """
    final_prob = prediction.final_probability
    home_won = outcome.get("home_won", False)
    actual = 1 if home_won else 0

    bs = brier_score(final_prob, actual)

    # Layer attribution: was each layer directionally helpful?
    base = prediction.base_probability
    sit = prediction.situational_adjustment
    info = prediction.information_edge

    base_correct = (base > 0.5 and home_won) or (base < 0.5 and not home_won)
    sit_helped = (sit > 0 and home_won) or (sit < 0 and not home_won) or sit == 0
    info_helped = (info > 0 and home_won) or (info < 0 and not home_won) or info == 0

    return GradingRecord(
        prediction_id=prediction.prediction_id,
        game_id=prediction.game_id,
        home_team=outcome.get("home_team"),
        away_team=outcome.get("away_team"),
        predicted_home_prob=final_prob,
        home_won=home_won,
        home_score=outcome.get("home_score"),
        away_score=outcome.get("away_score"),
        brier_score=round(bs, 6),
        edge_type=prediction.edge_type,
        effective_edge=prediction.effective_edge,
        recommendation=prediction.recommendation,
        base_correct=base_correct,
        sit_helped=sit_helped,
        info_helped=info_helped,
        model_variant=prediction.model_variant,
    )


# ---------------------------------------------------------------------------
# Write to Supabase calibration table
# ---------------------------------------------------------------------------

def write_calibration(game_date: date, grades: list[dict]) -> None:
    """Write daily calibration entry to Supabase."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not grades:
        return

    brier_scores = [g["brier_score"] for g in grades]
    avg_brier = sum(brier_scores) / len(brier_scores)

    base_correct = sum(1 for g in grades if g["base_correct"])
    sit_helped = sum(1 for g in grades if g["sit_helped"])
    info_helped = sum(1 for g in grades if g["info_helped"])

    attribution = {
        "base_correct": base_correct,
        "base_total": len(grades),
        "situational_helped": sit_helped,
        "information_helped": info_helped,
        "grades": [
            {
                "game": f"{g['away_team']} @ {g['home_team']}",
                "predicted": round(g["predicted_home_prob"], 3),
                "actual": "home" if g["home_won"] else "away",
                "score": f"{g['home_score']}-{g['away_score']}",
                "brier": round(g["brier_score"], 4),
                "edge_type": g["edge_type"],
                "recommendation": g["recommendation"],
            }
            for g in grades
        ],
    }

    row = {
        "date": game_date.isoformat(),
        "trade_count": len(grades),
        "brier_score_rolling": round(avg_brier, 6),
        "attribution_breakdown": json.dumps(attribution),
    }

    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/calibration",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            json=row,
            timeout=10,
        )
        if resp.status_code not in (200, 201):
            logger.error("Calibration write failed: %s %s", resp.status_code, resp.text)
        else:
            logger.info("Calibration written: %s, %d games, Brier=%.4f", game_date, len(grades), avg_brier)
    except Exception as e:
        logger.error("Calibration write error: %s", e)


# ---------------------------------------------------------------------------
# Discord posting
# ---------------------------------------------------------------------------

def _get_home_away_bias(game_date: date, grades: list[dict]) -> list[str]:
    """Calculate home/away bias for today's BET signals and 7-day rolling."""
    bet_grades = [g for g in grades if g.get("recommendation") == "BET"]
    today_home = sum(1 for g in bet_grades if g["predicted_home_prob"] > 0.5)
    today_away = sum(1 for g in bet_grades if g["predicted_home_prob"] <= 0.5)
    today_total = today_home + today_away

    # Away signals that were PASS with negative edge (should-have-bet-away but no edge)
    away_no_edge = sum(
        1 for g in grades
        if g.get("recommendation") == "PASS"
        and g["predicted_home_prob"] <= 0.5
        and (g.get("effective_edge") or 0) < 0
    )

    lines = [
        "",
        "**Home/Away Bias**",
        f"Home bets: {today_home}/{today_total} ({today_home/today_total:.0%})" if today_total else "Home bets: 0",
        f"Away bets: {today_away}/{today_total} ({today_away/today_total:.0%})" if today_total else "Away bets: 0",
        f"Away PASS (negative edge): {away_no_edge}",
    ]

    # 7-day rolling from trades table + per-day home% for streak detection
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        try:
            week_start = (game_date - timedelta(days=6)).isoformat()
            resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/trades",
                headers={
                    "apikey": SUPABASE_SERVICE_ROLE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                },
                params={
                    "select": "team_picked,game_id,created_at,games(home_team)",
                    "created_at": f"gte.{week_start}T00:00:00",
                    "order": "created_at",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                trades = resp.json()
                rolling_home = sum(
                    1 for t in trades
                    if t.get("team_picked", "").lower().strip()
                    == (t.get("games", {}) or {}).get("home_team", "").lower().strip()
                )
                rolling_total = len(trades)
                if rolling_total:
                    lines.append(
                        f"7-day rolling: {rolling_home}/{rolling_total} home ({rolling_home/rolling_total:.0%})"
                    )

                # Check for 3-day consecutive >75% home streak
                from collections import defaultdict
                daily_counts = defaultdict(lambda: {"home": 0, "total": 0})
                for t in trades:
                    day = t.get("created_at", "")[:10]
                    daily_counts[day]["total"] += 1
                    if (t.get("team_picked", "").lower().strip()
                            == (t.get("games", {}) or {}).get("home_team", "").lower().strip()):
                        daily_counts[day]["home"] += 1

                # Check last 3 days with trades
                recent_days = sorted(daily_counts.keys(), reverse=True)[:3]
                if len(recent_days) >= 3:
                    streak = all(
                        daily_counts[d]["home"] / daily_counts[d]["total"] > 0.75
                        for d in recent_days
                        if daily_counts[d]["total"] > 0
                    )
                    if streak:
                        lines.append("RED FLAG: >75% home bets for 3 straight days — post-freeze fix needed")
                        logger.warning("HOME BIAS RED FLAG: >75%% home for 3 consecutive days")
        except Exception as e:
            logger.warning("Home/away bias lookup failed: %s", e)

    return lines


def _get_traded_summary(game_date: date, grades: list[dict]) -> list[str]:
    """Build traded-only performance section from settled trades."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return []

    date_start = f"{game_date.isoformat()}T00:00:00"
    date_end = f"{(game_date + timedelta(days=1)).isoformat()}T12:00:00"
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            },
            params=[
                ("select", "id,game_id,team_picked,amount_usdc,entry_price,outcome,pnl"),
                ("created_at", f"gte.{date_start}"),
                ("created_at", f"lt.{date_end}"),
                ("order", "id"),
            ],
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        trades = resp.json()
    except Exception:
        return []

    if not trades:
        return []

    wins = [t for t in trades if t.get("outcome") == "win"]
    losses = [t for t in trades if t.get("outcome") == "loss"]
    pending = [t for t in trades if t.get("outcome") == "pending"]

    lines = [
        "",
        f"**Traded — {len(wins)}W-{len(losses)}L" + (f"-{len(pending)}P" if pending else "") + f" ({len(trades)} trades)**",
    ]

    for t in trades:
        outcome = t.get("outcome", "pending")
        mark = "W" if outcome == "win" else ("L" if outcome == "loss" else "?")
        lines.append(
            f"[{mark}] {t['team_picked']} @ {t['entry_price']}"
        )

    return lines


def post_performance(game_date: date, grades: list[dict]) -> bool:
    """Post performance summary to #performance."""
    if not grades:
        return False

    brier_scores = [g["brier_score"] for g in grades]
    avg_brier = sum(brier_scores) / len(brier_scores)
    correct = sum(1 for g in grades if (g["predicted_home_prob"] > 0.5) == g["home_won"])

    lines = [
        f"**Performance — {game_date.strftime('%A, %B %-d')}**",
        f"Games graded: {len(grades)}",
        f"Directional accuracy: {correct}/{len(grades)} ({correct/len(grades):.0%})",
        f"Average Brier: **{avg_brier:.4f}**",
        "",
    ]

    for g in grades:
        winner = g["home_team"] if g["home_won"] else g["away_team"]
        pick_correct = (g["predicted_home_prob"] > 0.5) == g["home_won"]
        mark = "+" if pick_correct else "-"
        edge_str = f" edge={g['effective_edge']:+.1%}" if g.get("effective_edge") else ""
        rec = g.get("recommendation", "")
        lines.append(
            f"[{mark}] {g['away_team']} @ {g['home_team']} — "
            f"{g['home_score']}-{g['away_score']} "
            f"(pred: {g['predicted_home_prob']:.0%} home, Brier: {g['brier_score']:.4f}"
            f"{edge_str} {rec})"
        )

    # Traded performance section
    lines.extend(_get_traded_summary(game_date, grades))

    # Home/away bias section
    lines.extend(_get_home_away_bias(game_date, grades))

    msg = "\n".join(lines)

    config_path = Path(__file__).resolve().parent.parent.parent / ".discord_config.json"
    if not config_path.exists():
        return False
    config = json.loads(config_path.read_text())

    channel_id = config.get("channels", {}).get("performance")
    bot_token = config.get("tokens", {}).get("performance")
    if not channel_id or not bot_token:
        return False

    try:
        resp = requests.post(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            headers={"Authorization": f"Bot {bot_token}", "Content-Type": "application/json"},
            json={"content": msg[:2000]},
            timeout=10,
        )
        return resp.status_code in (200, 201)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Lessons generation
# ---------------------------------------------------------------------------

LESSONS_PATH = Path(__file__).resolve().parent.parent.parent / "tasks" / "lessons.md"

# Brier thresholds for lesson triggers
BAD_BRIER = 0.35       # single-game Brier that warrants investigation
HIGH_EDGE_MISS = 0.10  # missed BET with effective_edge >= this


def _detect_patterns(grades: list[dict], game_date: date) -> list[dict]:
    """Detect actionable patterns from grading results.

    Returns list of {pattern, rule, context} dicts.
    """
    lessons = []

    brier_scores = [g["brier_score"] for g in grades]
    avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0
    correct = sum(1 for g in grades if (g["predicted_home_prob"] > 0.5) == g["home_won"])

    # Pattern 1: High-Brier misses on BET recommendations
    bad_bets = [
        g for g in grades
        if g["recommendation"] == "BET"
        and g["brier_score"] >= BAD_BRIER
        and (g["predicted_home_prob"] > 0.5) != g["home_won"]
    ]
    for g in bad_bets:
        edge = g.get("effective_edge") or 0
        if abs(edge) >= HIGH_EDGE_MISS:
            winner = g["home_team"] if g["home_won"] else g["away_team"]
            lessons.append({
                "pattern": (
                    f"BET signal on {g['away_team']} @ {g['home_team']} with "
                    f"edge {edge:+.1%} missed — {winner} won "
                    f"{g['home_score']}-{g['away_score']}. "
                    f"Brier: {g['brier_score']:.4f}."
                ),
                "rule": (
                    f"Edges >{abs(edge):.0%} on {'high' if abs(edge) > 0.15 else 'moderate'}-confidence "
                    f"calls need scrutiny. Check if base probability or Layer 3 data was stale."
                ),
                "context": "Both",
            })

    # Pattern 2: Base probability wrong direction
    base_wrong = [g for g in grades if not g["base_correct"]]
    if len(base_wrong) > len(grades) * 0.4 and len(grades) >= 5:
        teams = [f"{g['away_team']}@{g['home_team']}" for g in base_wrong[:3]]
        lessons.append({
            "pattern": (
                f"Base probability wrong direction in {len(base_wrong)}/{len(grades)} games "
                f"({', '.join(teams)}...)."
            ),
            "rule": "Review win% data freshness and split record extraction. Base should be directionally correct >55% of the time.",
            "context": "Both",
        })

    # Pattern 3: Situational adjustments consistently unhelpful
    sit_hurt = [g for g in grades if not g["sit_helped"]]
    if len(sit_hurt) > len(grades) * 0.5 and len(grades) >= 5:
        lessons.append({
            "pattern": f"Situational adjustments hurt predictions in {len(sit_hurt)}/{len(grades)} games.",
            "rule": "Review form weighting and B2B penalty magnitudes. If >50% of sit adjustments are counterproductive, reduce their caps.",
            "context": "Both",
        })

    # Pattern 4: Day-level Brier above threshold
    if avg_brier > 0.25 and len(grades) >= 5:
        lessons.append({
            "pattern": f"Daily Brier {avg_brier:.4f} above 0.25 threshold on {len(grades)} games ({correct}/{len(grades)} correct).",
            "rule": "Investigate whether edge inflation or missing data drove the miss. Check if splits and form data were populated.",
            "context": "Both",
        })

    return lessons


def write_lessons(new_lessons: list[dict], game_date: date) -> int:
    """Append new lessons to tasks/lessons.md. Returns count written."""
    if not new_lessons:
        return 0

    if not LESSONS_PATH.exists():
        return 0

    existing = LESSONS_PATH.read_text()

    # Find the next lesson number
    import re
    numbers = re.findall(r"### Lesson (\d+)", existing)
    next_num = max((int(n) for n in numbers), default=0) + 1

    entries = []
    for lesson in new_lessons:
        entry = (
            f"\n### Lesson {next_num}\n"
            f"- **Pattern:** {lesson['pattern']}\n"
            f"- **Rule:** {lesson['rule']}\n"
            f"- **Context:** {lesson['context']}\n"
            f"- **Date:** {game_date.isoformat()}\n"
        )
        entries.append(entry)
        next_num += 1

    LESSONS_PATH.write_text(existing + "\n".join(entries))
    logger.info("Wrote %d lessons to %s", len(entries), LESSONS_PATH)
    return len(entries)


# ---------------------------------------------------------------------------
# Main grading pipeline
# ---------------------------------------------------------------------------

def grade_date(game_date: date | None = None) -> list[dict]:
    """Grade all resolved games for a date.

    Default: yesterday (games that have finished).
    """
    if game_date is None:
        game_date = date.today() - timedelta(days=1)

    logger.info("Grading games for %s", game_date.isoformat())

    # Get outcomes
    nba_outcomes = get_resolved_nba(game_date)
    nhl_outcomes = get_resolved_nhl(game_date)
    all_outcomes = nba_outcomes + nhl_outcomes
    logger.info("Resolved: %d NBA, %d NHL", len(nba_outcomes), len(nhl_outcomes))

    if not all_outcomes:
        logger.info("No resolved games found for %s", game_date)
        return []

    # Get predictions from Supabase
    # Get games for team name matching
    games = get_games_from_supabase(game_date)
    game_map = {g["id"]: g for g in games}
    predictions = parse_prediction_rows(get_predictions(game_date), game_map)
    logger.info("Predictions found: %d", len(predictions))

    # Deduplicate predictions — keep first per (sport, home_team, away_team)
    seen_matchups = set()
    deduped_predictions = []
    for pred in predictions:
        key = (pred.sport, pred.home_team, pred.away_team)
        if key not in seen_matchups and key[1]:  # skip empty keys
            seen_matchups.add(key)
            deduped_predictions.append(pred)
    logger.info("Predictions after dedup: %d (was %d)", len(deduped_predictions), len(predictions))

    # Match predictions to outcomes (fuzzy: last word of team name)
    def _fuzzy_match(name_a: str, name_b: str) -> bool:
        if name_a == name_b:
            return True
        if not name_a or not name_b:
            return False
        return name_a.split()[-1].lower() == name_b.split()[-1].lower()

    grades = []
    graded_pairs: list[tuple[PredictionRecord, GradingRecord]] = []
    for pred in deduped_predictions:
        pred_home = pred.home_team
        pred_away = pred.away_team

        for outcome in all_outcomes:
            if _fuzzy_match(outcome["home_team"], pred_home) and _fuzzy_match(outcome["away_team"], pred_away):
                grade = grade_game(pred, outcome)
                if grade:
                    graded_pairs.append((pred, grade))
                    grades.append(grade.to_dict())
                break

    logger.info("Graded: %d/%d predictions matched to outcomes", len(grades), len(deduped_predictions))

    if grades:
        # Write to Supabase
        write_calibration(game_date, grades)

        stored_training = persist_training_examples(graded_pairs)
        if stored_training:
            logger.info("Ensemble training rows stored: %d", stored_training)

        # Post to Discord
        posted = post_performance(game_date, grades)
        logger.info("Performance posted to Discord: %s", posted)

        # Generate lessons from patterns
        new_lessons = _detect_patterns(grades, game_date)
        if new_lessons:
            written = write_lessons(new_lessons, game_date)
            logger.info("Lessons generated: %d", written)

        # Settle pending trades
        try:
            from src.trading.settler import settle_all
            game_results = {}
            for g in grades:
                gid = g.get("game_id")
                if gid:
                    game_results[gid] = g["home_won"]
            if game_results:
                settled = settle_all(game_results)
                if settled:
                    wins = sum(1 for s in settled if s.get("outcome") == "win")
                    losses = len(settled) - wins
                    pnl = sum(s.get("pnl", 0) for s in settled)
                    logger.info("Trades settled: %dW-%dL | P&L: $%.2f", wins, losses, pnl)
        except Exception as e:
            logger.warning("Trade settlement failed: %s", e)

        # Circuit breaker — record daily performance
        try:
            from src.model.safety import get_breaker
            brier_scores = [g["brier_score"] for g in grades if g.get("brier_score") is not None]
            if brier_scores:
                day_brier = sum(brier_scores) / len(brier_scores)
                # CLV avg: use 0 until CLV data accumulates
                clv_avg = 0.0
                try:
                    from src.model.clv import check_clv_health
                    clv_health = check_clv_health()
                    if clv_health["sample_size"] >= 5:
                        clv_avg = clv_health["rolling_7d_clv_bps"] / 100  # bps → %
                except Exception:
                    pass
                get_breaker().record_day(game_date, day_brier, clv_avg)
                logger.info("Circuit breaker recorded: Brier=%.3f, CLV=%.1f%%", day_brier, clv_avg)
        except Exception as e:
            logger.warning("Circuit breaker recording failed: %s", e)

        # Backfill CLV results
        try:
            from src.model.clv import backfill_results
            for g in grades:
                gid = g.get("game_id")
                if gid:
                    result = "win" if g["home_won"] else "loss"
                    backfill_results(gid, result)
        except Exception as e:
            logger.warning("CLV result backfill failed: %s", e)

    return grades


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) > 1:
        # Grade a specific date
        target = date.fromisoformat(sys.argv[1])
    else:
        # Default: yesterday
        target = date.today() - timedelta(days=1)

    grades = grade_date(target)
    print(f"\nGraded {len(grades)} games for {target}")
    for g in grades:
        mark = "+" if (g["predicted_home_prob"] > 0.5) == g["home_won"] else "-"
        print(f"  [{mark}] {g['away_team']} @ {g['home_team']}: "
              f"pred={g['predicted_home_prob']:.0%} actual={'home' if g['home_won'] else 'away'} "
              f"Brier={g['brier_score']:.4f}")
