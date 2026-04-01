#!/usr/bin/env python3
"""
Backtest Analysis — calibration report from backtest CSV.

Zero Supabase dependency. Reads only from CSV output.

Usage:
    python scripts/backtest_analysis.py backtest_results/nba_nhl_backtest_*.csv
    python scripts/backtest_analysis.py backtest_results/nba_nhl_backtest_*.csv --sport nba
    python scripts/backtest_analysis.py backtest_results/nba_nhl_backtest_*.csv --sensitivity
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Model functions (duplicated from runner to keep this standalone)
# ---------------------------------------------------------------------------

def logit(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return math.log(p / (1 - p))


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(paths: list[str], sport_filter: str | None = None) -> list[dict]:
    """Load and merge rows from one or more CSV files."""
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if sport_filter and row["sport"] != sport_filter:
                    continue
                # Parse numeric fields
                for key in ("base_prob_home", "situational_adj", "final_prob_home", "brier_score",
                            "home_win_rate", "away_win_rate", "home_strength", "away_strength",
                            "shrinkage_factor"):
                    if row.get(key) and row[key] != "None" and row[key] != "":
                        row[key] = float(row[key])
                    else:
                        row[key] = None
                for key in ("home_wins", "home_losses", "away_wins", "away_losses",
                            "home_score", "away_score", "home_rest_days", "away_rest_days"):
                    if row.get(key) and row[key] != "None" and row[key] != "":
                        row[key] = int(row[key])
                    else:
                        row[key] = None
                for key in ("home_b2b", "away_b2b", "actual_home_win"):
                    if row.get(key) and row[key] != "None" and row[key] != "":
                        row[key] = row[key] == "True"
                    else:
                        row[key] = None
                rows.append(row)
    return rows


def graded_rows(rows: list[dict]) -> list[dict]:
    """Filter to rows with actual outcomes."""
    return [r for r in rows if r.get("brier_score") is not None and r.get("actual_home_win") is not None]


# ---------------------------------------------------------------------------
# Section 1: Overall Brier Score
# ---------------------------------------------------------------------------

def section_overall(rows: list[dict]):
    print("=" * 70)
    print("1. OVERALL BRIER SCORE")
    print("=" * 70)
    print(f"   Reference: random=0.2500 | decent model~0.22 | good~0.20")
    print()

    graded = graded_rows(rows)
    if not graded:
        print("   No graded games found.\n")
        return

    by_sport = defaultdict(list)
    for r in graded:
        by_sport[r["sport"]].append(r["brier_score"])

    total_sum = 0.0
    total_n = 0
    for sport in sorted(by_sport):
        scores = by_sport[sport]
        mean = sum(scores) / len(scores)
        total_sum += sum(scores)
        total_n += len(scores)
        print(f"   {sport.upper():4s}:  {len(scores):5d} games  |  mean Brier: {mean:.4f}")

    if total_n:
        print(f"   {'ALL':4s}:  {total_n:5d} games  |  mean Brier: {total_sum / total_n:.4f}")
    print()


# ---------------------------------------------------------------------------
# Section 2: Calibration Table
# ---------------------------------------------------------------------------

def section_calibration(rows: list[dict]):
    print("=" * 70)
    print("2. CALIBRATION TABLE (10 buckets)")
    print("=" * 70)
    print(f"   {'Bucket':>10s}  {'Count':>6s}  {'Avg Pred':>9s}  {'Act Win%':>9s}  {'Bias':>7s}")
    print(f"   {'-' * 10}  {'-' * 6}  {'-' * 9}  {'-' * 9}  {'-' * 7}")

    graded = graded_rows(rows)
    buckets = defaultdict(lambda: {"preds": [], "outcomes": []})

    for r in graded:
        prob = r["final_prob_home"]
        bucket_idx = min(int(prob * 10), 9)
        label = f"{bucket_idx * 10}-{bucket_idx * 10 + 9}%"
        buckets[bucket_idx]["preds"].append(prob)
        buckets[bucket_idx]["outcomes"].append(int(r["actual_home_win"]))

    for i in range(10):
        label = f"{i * 10}-{i * 10 + 9}%"
        b = buckets[i]
        if not b["preds"]:
            print(f"   {label:>10s}  {0:>6d}  {'---':>9s}  {'---':>9s}  {'---':>7s}")
            continue
        n = len(b["preds"])
        avg_pred = sum(b["preds"]) / n
        act_rate = sum(b["outcomes"]) / n
        bias = act_rate - avg_pred
        print(f"   {label:>10s}  {n:>6d}  {avg_pred:>9.3f}  {act_rate:>9.3f}  {bias:>+7.3f}")

    print()


# ---------------------------------------------------------------------------
# Section 3: Layer Attribution
# ---------------------------------------------------------------------------

def section_attribution(rows: list[dict]):
    print("=" * 70)
    print("3. LAYER ATTRIBUTION (does Layer 2 help?)")
    print("=" * 70)

    graded = graded_rows(rows)
    if not graded:
        print("   No graded games.\n")
        return

    by_sport = defaultdict(lambda: {"base_brier": [], "final_brier": []})

    for r in graded:
        base = r["base_prob_home"]
        final = r["final_prob_home"]
        outcome = int(r["actual_home_win"])
        by_sport[r["sport"]]["base_brier"].append((base - outcome) ** 2)
        by_sport[r["sport"]]["final_brier"].append((final - outcome) ** 2)

    print(f"   {'Sport':>6s}  {'Layer1 Brier':>13s}  {'L1+L2 Brier':>12s}  {'Delta':>8s}  {'Verdict':>10s}")
    print(f"   {'-' * 6}  {'-' * 13}  {'-' * 12}  {'-' * 8}  {'-' * 10}")

    for sport in sorted(by_sport):
        d = by_sport[sport]
        base_mean = sum(d["base_brier"]) / len(d["base_brier"])
        final_mean = sum(d["final_brier"]) / len(d["final_brier"])
        delta = final_mean - base_mean
        verdict = "HELPS" if delta < -0.0005 else ("HURTS" if delta > 0.0005 else "NEUTRAL")
        print(f"   {sport.upper():>6s}  {base_mean:>13.4f}  {final_mean:>12.4f}  {delta:>+8.4f}  {verdict:>10s}")

    print()


# ---------------------------------------------------------------------------
# Section 4: Situational Breakdown
# ---------------------------------------------------------------------------

def section_situational(rows: list[dict]):
    print("=" * 70)
    print("4. SITUATIONAL BREAKDOWN (B2B effects)")
    print("=" * 70)

    graded = graded_rows(rows)
    cats = {
        "home_b2b_only": [],
        "away_b2b_only": [],
        "both_b2b": [],
        "neither_b2b": [],
    }

    for r in graded:
        hb = r["home_b2b"]
        ab = r["away_b2b"]
        if hb and ab:
            cats["both_b2b"].append(r)
        elif hb and not ab:
            cats["home_b2b_only"].append(r)
        elif ab and not hb:
            cats["away_b2b_only"].append(r)
        else:
            cats["neither_b2b"].append(r)

    print(f"   {'Category':>16s}  {'Count':>6s}  {'Mean Brier':>11s}  {'Home Win%':>10s}")
    print(f"   {'-' * 16}  {'-' * 6}  {'-' * 11}  {'-' * 10}")

    for cat_name, cat_rows in cats.items():
        if not cat_rows:
            print(f"   {cat_name:>16s}  {0:>6d}  {'---':>11s}  {'---':>10s}")
            continue
        n = len(cat_rows)
        mean_b = sum(r["brier_score"] for r in cat_rows) / n
        hw = sum(int(r["actual_home_win"]) for r in cat_rows) / n
        print(f"   {cat_name:>16s}  {n:>6d}  {mean_b:>11.4f}  {hw:>10.3f}")

    print()


# ---------------------------------------------------------------------------
# Section 5: Triage Tier Breakdown
# ---------------------------------------------------------------------------

def section_triage(rows: list[dict]):
    print("=" * 70)
    print("5. TRIAGE TIER BREAKDOWN")
    print("=" * 70)

    graded = graded_rows(rows)
    tiers = defaultdict(list)
    for r in graded:
        tiers[r.get("triage_tier", "unknown")].append(r)

    print(f"   {'Tier':>10s}  {'Count':>6s}  {'Mean Brier':>11s}  {'Accuracy':>9s}")
    print(f"   {'-' * 10}  {'-' * 6}  {'-' * 11}  {'-' * 9}")

    for tier in ("skip", "standard", "deep"):
        tier_rows = tiers.get(tier, [])
        if not tier_rows:
            print(f"   {tier:>10s}  {0:>6d}  {'---':>11s}  {'---':>9s}")
            continue
        n = len(tier_rows)
        mean_b = sum(r["brier_score"] for r in tier_rows) / n
        # Accuracy: did the predicted favourite win?
        correct = 0
        for r in tier_rows:
            pred_home = r["final_prob_home"] >= 0.5
            actual_home = r["actual_home_win"]
            if pred_home == actual_home:
                correct += 1
        acc = correct / n
        print(f"   {tier:>10s}  {n:>6d}  {mean_b:>11.4f}  {acc:>9.1%}")

    print()


# ---------------------------------------------------------------------------
# Section 6: Monthly Trend
# ---------------------------------------------------------------------------

def section_monthly(rows: list[dict]):
    print("=" * 70)
    print("6. MONTHLY TREND")
    print("=" * 70)

    graded = graded_rows(rows)
    months = defaultdict(list)
    for r in graded:
        ym = r["game_date"][:7]  # YYYY-MM
        months[ym].append(r["brier_score"])

    print(f"   {'Month':>8s}  {'Count':>6s}  {'Mean Brier':>11s}")
    print(f"   {'-' * 8}  {'-' * 6}  {'-' * 11}")

    for ym in sorted(months):
        scores = months[ym]
        mean = sum(scores) / len(scores)
        print(f"   {ym:>8s}  {len(scores):>6d}  {mean:>11.4f}")

    print()


# ---------------------------------------------------------------------------
# Section 7: Parameter Sensitivity (optional)
# ---------------------------------------------------------------------------

def section_sensitivity(rows: list[dict]):
    print("=" * 70)
    print("7. PARAMETER SENSITIVITY SWEEP")
    print("=" * 70)

    graded = graded_rows(rows)
    if not graded:
        print("   No graded games.\n")
        return

    b2b_vals = [0.05, 0.10, 0.14, 0.18, 0.25, 0.30]
    rest_vals = [0.02, 0.04, 0.06, 0.08, 0.10]

    results = []

    for b2b_pen in b2b_vals:
        for rest_per in rest_vals:
            total_brier = 0.0
            count = 0
            for r in graded:
                base = r["base_prob_home"]
                sport = r["sport"]
                hb = r["home_b2b"]
                ab = r["away_b2b"]
                hr = r["home_rest_days"]
                ar = r["away_rest_days"]
                outcome = int(r["actual_home_win"])

                # Recompute Layer 2 with swept params
                adj = 0.0
                if hb and not ab:
                    adj -= b2b_pen
                elif ab and not hb:
                    adj += b2b_pen

                def extra(rest):
                    return min(max((rest or 1) - 1, 0), 3)
                adj += (extra(hr) - extra(ar)) * rest_per

                final = logistic(logit(base) + adj)
                final = max(0.05, min(0.95, final))
                total_brier += (final - outcome) ** 2
                count += 1

            mean_brier = total_brier / count if count else 999
            results.append((mean_brier, b2b_pen, rest_per))

    results.sort()

    print(f"   {'Rank':>4s}  {'B2B Pen':>8s}  {'Rest/Day':>9s}  {'Mean Brier':>11s}")
    print(f"   {'-' * 4}  {'-' * 8}  {'-' * 9}  {'-' * 11}")

    for i, (mb, b2b, rest) in enumerate(results[:10]):
        print(f"   {i + 1:>4d}  {b2b:>8.2f}  {rest:>9.2f}  {mb:>11.4f}")

    # Also show per-sport breakdown for the best combo
    best_b2b, best_rest = results[0][1], results[0][2]
    print(f"\n   Best combo: B2B={best_b2b:.2f}, Rest/Day={best_rest:.2f}")
    for sport_name in ("nba", "nhl"):
        sport_rows = [r for r in graded if r["sport"] == sport_name]
        if not sport_rows:
            continue
        total = 0.0
        for r in sport_rows:
            adj = 0.0
            if r["home_b2b"] and not r["away_b2b"]:
                adj -= best_b2b
            elif r["away_b2b"] and not r["home_b2b"]:
                adj += best_b2b

            def extra(rest):
                return min(max((rest or 1) - 1, 0), 3)
            adj += (extra(r["home_rest_days"]) - extra(r["away_rest_days"])) * best_rest
            final = logistic(logit(r["base_prob_home"]) + adj)
            final = max(0.05, min(0.95, final))
            total += (final - int(r["actual_home_win"])) ** 2
        print(f"   {sport_name.upper()}: {total / len(sport_rows):.4f} ({len(sport_rows)} games)")

    print()


# ---------------------------------------------------------------------------
# Section 8: Shrinkage Sweep (optional, requires home_strength/away_strength)
# ---------------------------------------------------------------------------

HOME_BONUS = {"nba": 0.363, "nhl": 0.200}

def section_shrinkage(rows: list[dict]):
    print("=" * 70)
    print("8. SHRINKAGE SWEEP")
    print("=" * 70)

    graded = graded_rows(rows)
    # Check if home_strength column exists (v2 CSV)
    has_strength = any(r.get("home_strength") is not None for r in graded)
    if not has_strength:
        print("   home_strength/away_strength columns not found — skipping.")
        print("   (Re-run backtest with v2 runner to generate these columns.)")
        print()
        return

    shrinkage_vals = [0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95, 1.00]

    for sport_name in ("nba", "nhl"):
        sport_rows = [r for r in graded if r["sport"] == sport_name
                      and r.get("home_strength") is not None
                      and r.get("away_strength") is not None]
        if not sport_rows:
            continue

        bonus = HOME_BONUS[sport_name]
        print(f"\n   SHRINKAGE SWEEP — {sport_name.upper()} ({len(sport_rows)} games)")
        print(f"   {'Shrinkage':>10s}  {'Mean Brier':>11s}  {'Worst Bias':>11s}")
        print(f"   {'-' * 10}  {'-' * 11}  {'-' * 11}")

        best_brier = 999.0
        best_shrink = 1.0
        results = []

        for shrink in shrinkage_vals:
            total_brier = 0.0
            # Calibration buckets for worst-bias calc
            buckets = defaultdict(lambda: {"preds": [], "outcomes": []})

            for r in sport_rows:
                hs = r["home_strength"]
                aws = r["away_strength"]
                outcome = int(r["actual_home_win"])

                raw_lo = logit(hs) - logit(aws) + bonus
                shrunk_lo = raw_lo * shrink

                # Re-apply Layer 2 from stored adj
                adj = r["situational_adj"] if r["situational_adj"] is not None else 0.0
                final = logistic(shrunk_lo + adj)
                final = max(0.05, min(0.95, final))

                total_brier += (final - outcome) ** 2

                bucket_idx = min(int(final * 10), 9)
                buckets[bucket_idx]["preds"].append(final)
                buckets[bucket_idx]["outcomes"].append(outcome)

            mean_brier = total_brier / len(sport_rows)

            # Worst bucket bias (max absolute bias across non-empty buckets)
            worst_bias = 0.0
            for i in range(10):
                b = buckets[i]
                if len(b["preds"]) >= 5:  # need enough samples
                    avg_pred = sum(b["preds"]) / len(b["preds"])
                    act_rate = sum(b["outcomes"]) / len(b["outcomes"])
                    bias = abs(act_rate - avg_pred)
                    worst_bias = max(worst_bias, bias)

            results.append((shrink, mean_brier, worst_bias))
            if mean_brier < best_brier:
                best_brier = mean_brier
                best_shrink = shrink

        for shrink, mb, wb in results:
            marker = " ← recommended" if shrink == best_shrink else ""
            print(f"   {shrink:>10.2f}  {mb:>11.4f}  {wb:>11.3f}{marker}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest calibration analysis")
    parser.add_argument("csv_files", nargs="+", help="CSV file(s) to analyse")
    parser.add_argument("--sport", type=str, default=None, help="Filter to sport (nba/nhl)")
    parser.add_argument("--sensitivity", action="store_true", help="Run parameter sensitivity sweep")
    args = parser.parse_args()

    rows = load_csv(args.csv_files, args.sport)
    graded = graded_rows(rows)

    print()
    print(f"Loaded {len(rows)} rows ({len(graded)} graded) from {len(args.csv_files)} file(s)")
    if args.sport:
        print(f"Filtered to sport: {args.sport}")
    print()

    section_overall(rows)
    section_calibration(rows)
    section_attribution(rows)
    section_situational(rows)
    section_triage(rows)
    section_monthly(rows)

    if args.sensitivity:
        section_sensitivity(rows)
        section_shrinkage(rows)


if __name__ == "__main__":
    main()
