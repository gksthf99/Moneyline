# Lessons — Polymarket Prediction System

Every agent reads this file at session start. Each entry is a rule learned from prediction outcomes.

---

## Entry Format

- **Pattern:** What went wrong or what was observed
- **Rule:** What to do instead (concrete, testable)
- **Context:** NBA / NHL / both
- **Date:** When the pattern was identified

---

## Placeholder Entries (replace with real lessons during Phase 0 dry-run)

### Lesson 1
- **Pattern:** Base-only Brier on 13 historical games was 0.28 — expected without situational/info data.
- **Rule:** Never evaluate model quality from base probability alone. All three layers must be active before drawing calibration conclusions.
- **Context:** Both
- **Date:** 2026-03-12

### Lesson 2
- **Pattern:** Marquee game edges (>8%) in back-testing consistently reverted — sharp money corrects fast.
- **Rule:** Treat large perceived edges on marquee games as specification errors. Investigate the model, do not trade.
- **Context:** Both
- **Date:** 2026-03-12

### Lesson 3
- **Pattern:** BET signal on Denver Nuggets @ Memphis Grizzlies with edge +16.3% missed — Memphis Grizzlies won 110-102. Brier: 0.5057.
- **Rule:** Edges >16% on high-confidence calls need scrutiny. Check if base probability or Layer 3 data was stale.
- **Context:** Both
- **Date:** 2026-03-18


### Lesson 4
- **Pattern:** BET signal on New Jersey Devils @ New York Rangers with edge +15.7% missed — New Jersey Devils won 3-6. Brier: 0.3866.
- **Rule:** Edges >16% on high-confidence calls need scrutiny. Check if base probability or Layer 3 data was stale.
- **Context:** Both
- **Date:** 2026-03-18

### Lesson 5
- **Pattern:** BET signal on Denver Nuggets @ Memphis Grizzlies with edge +16.3% missed — Memphis Grizzlies won 125-118. Brier: 0.5057.
- **Rule:** Edges >16% on high-confidence calls need scrutiny. Check if base probability or Layer 3 data was stale.
- **Context:** Both
- **Date:** 2026-03-18


### Lesson 6
- **Pattern:** BET signal on New Jersey Devils @ New York Rangers with edge +15.7% missed — New Jersey Devils won 3-6. Brier: 0.3866.
- **Rule:** Edges >16% on high-confidence calls need scrutiny. Check if base probability or Layer 3 data was stale.
- **Context:** Both
- **Date:** 2026-03-18

### Lesson 7
- **Pattern:** BET signal on Portland Trail Blazers @ Minnesota Timberwolves with edge +10.1% missed — Portland Trail Blazers won 104-108. Brier: 0.4306.
- **Rule:** Edges >10% on moderate-confidence calls need scrutiny. Check if base probability or Layer 3 data was stale.
- **Context:** Both
- **Date:** 2026-03-20

### Lesson 8 — RESOLVED
- **Pattern:** 5 of 6 NHL misses (Mar 18–20) were away team upsets where the model picked the home team. All failures clustered in the 50–65% probability band. Home-ice advantage (+25 Elo bonus) may be overweighting home teams when the away team is simply better. The bonus flattens meaningful skill gaps into coin-flip territory.
- **Rule:** Do not change the model yet — collect more data. Track NHL home-team misses specifically. If the pattern holds after 50+ NHL graded games, reduce the NHL home-ice bonus (currently +25) or make it conditional on relative team strength (e.g., suppress bonus when away team's points% exceeds home team's by >5%).
- **Context:** NHL
- **Date:** 2026-03-21
- **Fix implemented 2026-03-22:** Conditional suppression added to `nhl_base_probability()` in `src/model/baseline.py`. Bonus suppressed when `away_blended_pct - home_blended_pct > 0.05`. `HOME_ICE_BONUS = 25`, `SUPPRESSION_THRESHOLD = 0.05` as tunable constants. Logs `home_ice_suppressed: bool` for calibration tracking. 7 new tests passing.
- **Validation plan:** Shadow-run from 2026-03-22 until 50 NHL games graded, then compare suppressed vs unsuppressed Brier on the affected probability band (51–65%).

### Lesson 9
- **Pattern:** BET signal on Dallas Stars @ Minnesota Wild with edge +17.3% missed — Minnesota Wild won 2-1. Brier: 0.3715.
- **Rule:** Edges >17% on high-confidence calls need scrutiny. Check if base probability or Layer 3 data was stale.
- **Context:** Both
- **Date:** 2026-03-21

### Lesson 10
- **Pattern:** BET signal on Vegas Golden Knights @ Dallas Stars with edge +19.4% missed — Vegas Golden Knights won 2-3. Brier: 0.5756.
- **Rule:** Edges >19% on high-confidence calls need scrutiny. Check if base probability or Layer 3 data was stale.
- **Context:** Both
- **Date:** 2026-03-22


### Lesson 11 — RESOLVED
- **Pattern:** Situational adjustments hurt predictions in 9/14 games.
- **Rule:** Review form weighting and B2B penalty magnitudes. If >50% of sit adjustments are counterproductive, reduce their caps.
- **Context:** Both
- **Date:** 2026-03-22
- **Fix implemented 2026-03-23:** NHL total sit cap (±5%) added to `nhl_situational()`. NHL form cap reduced from ±5% to ±3%. Individual factors keep their caps but the combined sum is now clamped. On the Stars game, this would have reduced the sit adj from +9.6% to +5.0% (and form from +5% to +3%). Backtest confirmed sit adj provides only 0.0009 Brier improvement on average over 1,176 NHL games — large adjustments are noise. NBA uncapped for now but flagged for monitoring (theoretical max ~±14%).

### Lesson 12 — RESOLVED
- **Pattern:** Daily Brier 0.2517 above 0.25 threshold on 14 games (8/14 correct).
- **Rule:** Investigate whether edge inflation or missing data drove the miss. Check if splits and form data were populated.
- **Context:** Both
- **Date:** 2026-03-22
- **Root cause identified 2026-03-23:** Not a Layer 1 calibration issue — full-season backtest (2,320 games) confirmed NHL shrinkage 1.00 is still optimal. The problem was Layer 2 compounding: form (+5%) + H2H (+3%) + goalie (+4%) stacked to +9.6% on Stars (Brier=0.5756). That single game swung the daily average by 0.025. Without it, combined Brier would have been 0.227. Fix: NHL sit cap + form cap reduction deployed.

### Lesson 13
- **Pattern:** BET signal on Toronto Maple Leafs @ Boston Bruins with edge +17.3% missed — Toronto Maple Leafs won 2-4. Brier: 0.6374.
- **Rule:** Edges >17% on high-confidence calls need scrutiny. Check if base probability or Layer 3 data was stale.
- **Context:** Both
- **Date:** 2026-03-24

### Lesson 14
- **Pattern:** Situational adjustments hurt predictions in 9/14 games.
- **Rule:** Review form weighting and B2B penalty magnitudes. If >50% of sit adjustments are counterproductive, reduce their caps.
- **Context:** Both
- **Date:** 2026-03-25

### Lesson 15
- **Pattern:** Situational adjustments hurt predictions in 10/16 games.
- **Rule:** Review form weighting and B2B penalty magnitudes. If >50% of sit adjustments are counterproductive, reduce their caps.
- **Context:** Both
- **Date:** 2026-03-26

### Lesson 16
- **Pattern:** A newly added calibration layer underperformed the legacy backtest probabilities on both full-season comparison runs, worsening both Brier and W/L.
- **Rule:** Do not ship or keep a new calibration layer unless it beats the legacy model on both Brier and W/L on the same historical sample. Architecture and data improvements can stay; model-layer additions require direct evidence.
- **Context:** Both
- **Date:** 2026-03-29
