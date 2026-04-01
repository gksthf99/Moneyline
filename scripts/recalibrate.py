"""
Rolling recalibration — re-optimize model parameters from recent results.

Reads the latest backtest CSV, re-runs shrinkage sweep, and updates
baseline.py if a better value is found.

Usage:
  python3 scripts/recalibrate.py                # full recalibration
  python3 scripts/recalibrate.py --dry-run      # show results without applying
  python3 scripts/recalibrate.py --last 30      # only use last 30 days

Intended to run weekly via cron:
  0 8 * * 0  cd /path/to/project && python3 scripts/recalibrate.py >> logs/recalibrate.log 2>&1
"""

import csv
import math
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent


def load_backtest_data(days_back: int | None = None) -> list[dict]:
    """Load the latest backtest CSV."""
    csv_dir = ROOT / "backtest_results"
    csvs = sorted(csv_dir.glob("nba_nhl_backtest_*.csv"))
    if not csvs:
        print("No backtest CSV found in backtest_results/")
        sys.exit(1)

    latest = csvs[-1]
    print(f"Loading: {latest.name}")

    rows = []
    with open(latest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if days_back:
                gd = row.get("game_date", "")
                if gd:
                    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
                    if gd < cutoff:
                        continue
            rows.append(row)

    print(f"Loaded {len(rows)} games" + (f" (last {days_back} days)" if days_back else ""))
    return rows


def shrinkage_sweep(rows: list[dict], sport: str) -> tuple[float, float]:
    """Find optimal shrinkage factor for a sport.

    Returns (best_shrinkage, best_brier).
    """
    sport_rows = [r for r in rows if r.get("sport", "").lower() == sport.lower()]
    if not sport_rows:
        return 1.0, 999.0

    # We need to recompute probabilities with different shrinkage values
    # Use the stored home_strength, away_strength to recompute
    factors = [round(x * 0.05 + 0.50, 2) for x in range(11)]  # 0.50 to 1.00

    home_bonus = {"nba": 0.363, "nhl": 0.200}.get(sport.lower(), 0.3)

    best_factor = 1.0
    best_brier = 999.0

    for factor in factors:
        brier_sum = 0.0
        count = 0

        for r in sport_rows:
            try:
                h_str = float(r.get("home_strength", 0.5))
                a_str = float(r.get("away_strength", 0.5))
                # actual_home_win is stored as True/False string in CSV
                actual_raw = str(r.get("actual_home_win", "")).lower()
                if actual_raw in ("true", "1"):
                    actual = 1.0
                elif actual_raw in ("false", "0"):
                    actual = 0.0
                else:
                    continue  # skip ungraded games
            except (ValueError, TypeError):
                continue

            # For NHL: respect home-ice suppression flag from v3 CSV
            bonus = home_bonus
            suppressed = r.get("home_ice_suppressed", "")
            if sport.lower() == "nhl" and str(suppressed).lower() in ("true", "1"):
                bonus = 0.0

            # Recompute probability with this shrinkage
            h_lo = math.log(h_str / (1 - h_str)) if 0 < h_str < 1 else 0
            a_lo = math.log(a_str / (1 - a_str)) if 0 < a_str < 1 else 0
            raw_lo = h_lo - a_lo + bonus
            shrunk_lo = raw_lo * factor
            prob = 1.0 / (1.0 + math.exp(-shrunk_lo))

            # Add H2H adjustment if present (probability space, not affected by shrinkage)
            h2h_adj = float(r.get("h2h_adj", 0) or 0)
            prob += h2h_adj
            prob = max(0.05, min(0.95, prob))

            brier = (prob - actual) ** 2
            brier_sum += brier
            count += 1

        if count > 0:
            mean_brier = brier_sum / count
            if mean_brier < best_brier:
                best_brier = mean_brier
                best_factor = factor

    return best_factor, best_brier


def read_current_shrinkage() -> dict:
    """Read current shrinkage values from baseline.py."""
    baseline_path = ROOT / "src" / "model" / "baseline.py"
    text = baseline_path.read_text()

    nba_match = re.search(r"NBA_SHRINKAGE\s*=\s*([\d.]+)", text)
    nhl_match = re.search(r"NHL_SHRINKAGE\s*=\s*([\d.]+)", text)

    return {
        "NBA": float(nba_match.group(1)) if nba_match else 0.70,
        "NHL": float(nhl_match.group(1)) if nhl_match else 1.00,
    }


def apply_shrinkage(sport: str, new_value: float):
    """Update shrinkage value in baseline.py."""
    baseline_path = ROOT / "src" / "model" / "baseline.py"
    text = baseline_path.read_text()

    key = f"{sport.upper()}_SHRINKAGE"
    pattern = rf"({key}\s*=\s*)[\d.]+"
    new_text = re.sub(pattern, rf"\g<1>{new_value:.2f}", text)

    baseline_path.write_text(new_text)
    print(f"  Updated {key} = {new_value:.2f} in baseline.py")


def main():
    dry_run = "--dry-run" in sys.argv
    days_back = None
    for i, arg in enumerate(sys.argv):
        if arg == "--last" and i + 1 < len(sys.argv):
            days_back = int(sys.argv[i + 1])

    rows = load_backtest_data(days_back)
    current = read_current_shrinkage()

    print(f"\nCurrent shrinkage: NBA={current['NBA']:.2f}, NHL={current['NHL']:.2f}")
    print(f"{'='*50}")

    changes = []

    for sport in ["NBA", "NHL"]:
        best_factor, best_brier = shrinkage_sweep(rows, sport)
        sport_rows = [r for r in rows if r.get("sport", "").lower() == sport.lower()]

        # Current Brier for comparison
        current_brier = 0.0
        count = 0
        for r in sport_rows:
            try:
                current_brier += float(r.get("brier_score", 0))
                count += 1
            except (ValueError, TypeError):
                pass
        current_brier = current_brier / count if count > 0 else 999

        improvement = current_brier - best_brier
        improved = improvement > 0.001  # meaningful improvement threshold

        print(f"\n{sport}:")
        print(f"  Current:  shrinkage={current[sport]:.2f}, Brier={current_brier:.4f}")
        print(f"  Optimal:  shrinkage={best_factor:.2f}, Brier={best_brier:.4f}")
        print(f"  Delta:    {improvement:+.4f} {'BETTER' if improved else '(no improvement)'}")

        if improved and best_factor != current[sport]:
            changes.append((sport, best_factor, improvement))

    print(f"\n{'='*50}")

    if not changes:
        print("No parameter changes needed. Model is well-calibrated.")
        return

    for sport, new_val, delta in changes:
        if dry_run:
            print(f"[DRY RUN] Would update {sport} shrinkage: {current[sport]:.2f} → {new_val:.2f} (Brier -{delta:.4f})")
        else:
            apply_shrinkage(sport, new_val)
            print(f"Applied {sport} shrinkage: {current[sport]:.2f} → {new_val:.2f} (Brier -{delta:.4f})")

    if not dry_run and changes:
        print(f"\nRecalibration complete. {len(changes)} parameter(s) updated.")
        print("Re-run the backtest to verify: python3 scripts/backtest_runner.py")


if __name__ == "__main__":
    main()
