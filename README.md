# Sports Polymarket

An automated sports prediction and trading system that identifies mispriced moneyline markets on [Polymarket](https://polymarket.com) for NBA and NHL games. The system runs a multi-layered probabilistic model, finds edges against market-implied probabilities, and executes trades via the Polymarket CLOB API.

**Status:** Shelved. The system ran live for ~2 weeks, grading 217 games at a **0.2078 Brier score** and settling 31 trades at an **18W-13L record** (58.1% win rate).

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [The Prediction Model](#the-prediction-model)
  - [Layer 1: Baseline Probability](#layer-1-baseline-probability)
  - [Layer 2: Situational Adjustments](#layer-2-situational-adjustments)
  - [Layer 3: Information Edge](#layer-3-information-edge)
  - [Combining Layers](#combining-layers)
- [Edge Detection & Position Sizing](#edge-detection--position-sizing)
- [Ensemble Model](#ensemble-model)
- [Safety & Risk Management](#safety--risk-management)
- [Data Sources](#data-sources)
- [Infrastructure](#infrastructure)
- [Results & Findings](#results--findings)
- [Project Structure](#project-structure)
- [Setup](#setup)

---

## How It Works

Every morning at 7:00 AM, the system:

1. **Pulls today's slate** of NBA and NHL games from ESPN and NHL APIs
2. **Triages** each game as deep-dive, standard, or skip based on situational edges (back-to-backs, travel, rest mismatches)
3. **Runs a 3-layer probability model** on each game, decomposing the prediction into base team quality, situational factors, and information edges (injuries/goalies)
4. **Compares model probability to Polymarket prices** to find mispriced markets
5. **Sizes positions** using fractional Kelly criterion
6. **Executes trades** 30 minutes before tip-off via a VPS in Finland (Polymarket geoblocks US IPs)
7. **Grades results** overnight against actual outcomes, calculates Brier scores, settles trades, and feeds lessons back into the model

Everything posts to Discord for monitoring: daily slates, research threads per game, trade signals, and nightly performance reports.

---

## Architecture

```
LOCAL MACHINE (US)                           VPS (Helsinki, Finland)
================================            ================================
                                             
  morning_slate.py  (7:00 AM)                vps_executor.py  (30m pre-tipoff)
  ├─ ESPN / NHL APIs                         ├─ Pulls BET signals from Supabase
  ├─ Triage & schedule                       ├─ Live price re-validation
  └─ Supabase + Discord                      ├─ Kelly sizing against live bankroll
                                             ├─ Polymarket CLOB order placement
  research_agent.py (7:30 AM)                └─ Discord trade alerts
  ├─ 3-layer probability model               
  ├─ Edge calculation                        clv_capture.py  (every 2m)
  ├─ Supabase signals                        └─ Closing price at tip-off
  └─ Discord research threads                
                                             redeem_positions.py  (daily)
  alert_agent.py    (pre-game)               └─ Batch redeem via Gnosis Safe
  ├─ Confirmed goalies                       
  ├─ Late injury updates                     settler.py  (6:30 AM)
  └─ Discord pre-game briefs                 └─ Grade W/L, calculate P&L
                                             
  performance_agent (overnight)              
  ├─ Match predictions to outcomes           
  ├─ Brier score calculation                 
  ├─ Lesson extraction                       
  └─ Discord performance report              
                                             
         ┌──────────────────┐                
         │    Supabase      │                
         │   (PostgreSQL)   │                
         │                  │                
         │  games           │                
         │  research        │                
         │  trades          │                
         │  calibration     │                
         │  clv             │                
         │  ensemble_training│               
         └──────────────────┘                
```

The local machine handles all analysis and writes BET signals to Supabase. The VPS only does execution. This separation keeps the model logic centralized while bypassing Polymarket's IP restrictions for order placement.

Communication between the two happens through Supabase (shared PostgreSQL) and Tailscale (private network for SSH).

---

## The Prediction Model

The model decomposes game predictions into three independent layers, each capturing a different type of signal.

### Layer 1: Baseline Probability

Two approaches depending on sport and data availability:

#### Elo Model (NBA fallback, NHL primary)

Converts team win percentages to Elo ratings, then computes expected win probability:

```
Elo = 1500 + 400 * log10(Win% / (1 - Win%))

P(Home) = 1 / (1 + 10^(-(EloHome - EloAway + HCA) * Shrinkage / 400))
```

| Parameter | NBA | NHL |
|-----------|-----|-----|
| Home Bonus | +30 Elo | +25 Elo |
| Shrinkage | 0.70 | 1.00 |

Shrinkage compresses overconfident predictions toward 50%. NBA needs more compression (0.70) because raw win% differences overstate true talent gaps. NHL's parity means no compression needed.

#### Net Rating Model (NBA primary)

Uses offensive/defensive ratings per 100 possessions from NBA.com:

```
Strength = 0.80 * SOS_Adjusted_NetRtg + 0.20 * Recent_Form_NetRtg

P(Home) = 1 / (1 + 10^(-(StrengthHome - StrengthAway + 2.5) * 0.65 / 16.0))
```

- **K = 16.0** is the logistic scaling constant (surface is flat around K=12-22, model is robust)
- **Shrinkage = 0.65** compresses net rating extremes
- **HCA = +2.5 pts** net rating home court advantage
- **SOS weighting (0.80)** favors strength-of-schedule-adjusted ratings over recent form

Validated on 1,131 NBA games (full season backtest): **Brier 0.2094** vs 0.2219 for the Elo model.

#### NHL Enhancements

NHL baseline blends multiple signals:

```
Adjusted_Strength = 0.70 * Points% + 0.30 * xGF%
```

Expected goals for percentage (xGF%) from Natural Stat Trick catches teams with unsustainable shooting/save percentages. A team winning on luck (high PDO) will have xGF% diverge from actual points%, signaling regression.

**Home ice suppression:** When the away team's strength exceeds home by >5%, the home ice bonus is zeroed out to prevent the bonus from flipping the predicted winner in mismatched games.

### Layer 2: Situational Adjustments

Game-context factors that shift probability from baseline:

| Factor | NBA | NHL | Source |
|--------|-----|-----|--------|
| Back-to-back penalty | 3.5% | 2.5% | Schedule analysis |
| Rest advantage | 1.2%/day (cap 3%) | 1.0%/day (cap 3%) | Days since last game |
| Travel fatigue | 1.0%/timezone | 1.2%/timezone | Timezone crossings |
| Recent form (L10) | Cap +/-5% | Cap +/-3% | Last 10 games |
| Head-to-head | Cap +/-3% | Cap +/-3% | Season series (min 2 meetings) |

**Form blending:**
```
Blended_Form = 0.60 * L10_Win% + 0.40 * Season_Win%
Adjustment = Clamp(Home_Blended - Away_Blended, -Cap, +Cap)
```

**H2H weighting by sample size:**
```
Weight = Min(1.0, H2H_Games / 4)
Adjustment = Clamp((H2H_Win% - 50%) * Weight, -3%, +3%)
```

NHL form cap is tighter (3% vs 5%) because hockey outcomes are more random -- a hot streak in the NHL is less predictive than in the NBA.

**NHL total situational cap:** All NHL situational adjustments are capped at +/-5% combined to prevent independent signals from compounding unrealistically.

**Situational weight:** The total Layer 2 adjustment is scaled by 0.25 (NBA), calibrated via backtest to improve Brier by ~0.0015 on average.

### Layer 3: Information Edge

Player availability impact, tiered by importance:

| Tier | Impact (when OUT) | Examples |
|------|-------------------|----------|
| Tier 1 (MVP-caliber) | 5-8% | Jokic, McDavid |
| Tier 2 (All-Star) | 3-5% | Top-line centers, All-NBA |
| Tier 3 (Quality starter) | 1-3% | Starting caliber players |
| Tier 4 (Role player) | 0.5-1% | Rotation pieces |

**Status weighting:**
```
OUT        = 1.00 (full impact)
DOUBTFUL   = 0.75
QUESTIONABLE = 0.50
PROBABLE   = 0.15
```

**Diminishing returns for stacked injuries:**
```
1st injury at tier = 100% of impact
2nd injury at tier = 50%
3rd+ at tier       = 25%
```

This prevents unrealistic compounding. If a team is missing 3 role players, the model doesn't treat it as a 3x penalty -- the marginal impact of each additional absence decreases because the replacement-level talent pool is already being used.

**Total information edge capped at +/-15%.**

### Combining Layers

```
Raw_Prob = Base_Probability + Situational_Adjustment + Information_Edge
Final_Prob = Clamp(Raw_Prob, 0.05, 0.95)
```

The 5-95% clamp prevents the model from ever expressing certainty, which would catastrophically skew Brier scores on upsets.

---

## Edge Detection & Position Sizing

### Finding Edge

The model compares its probability to the Polymarket ask price, adjusting for the bid-ask spread:

```
Vig = (Ask - Bid) / ((Ask + Bid) / 2)
True_Implied = Ask / (1 - Vig)
Effective_Edge = Model_Prob - True_Implied - 0.5%   (slippage cost)
```

**Two-sided edge calculation:** For each game, edge is computed for both home and away outcomes. The system always bets the model's predicted winner -- edge determines *whether* to bet, never *which side*.

**Sport-specific minimum thresholds:**

| Timeframe | NBA | NHL |
|-----------|-----|-----|
| Same-day (< 12h) | 1.5% | 4.0% |
| 24-48h out | 3.0% | 6.0% |
| 48h+ out | 5.0% | 8.0% |

NHL thresholds are much higher because the model's NHL accuracy (Brier 0.240) is weaker than NBA (Brier 0.176). Higher thresholds demand more edge to compensate for model uncertainty.

### Kelly Criterion Sizing

Position sizes are determined by the Kelly criterion for binary markets:

```
Kelly_Fraction = (Model_Prob - Ask) / (1 - Ask)
```

Where `(1 - Ask) / Ask` represents the net odds (profit per unit risked on a Polymarket binary that pays 1.00 on win).

**Fractional Kelly:** The system uses 1/4 Kelly for conservative sizing:

```
Position_Size = Bankroll * Kelly_Fraction * 0.25
```

Full Kelly is mathematically optimal for maximizing long-term growth rate, but the variance is extreme. Quarter Kelly sacrifices ~44% of the theoretical growth rate but reduces drawdowns by ~75%.

**Position caps by sport:**

| Sport | Max Position | Floor |
|-------|-------------|-------|
| NBA | 5% of bankroll | 1% |
| NHL | 2.5% of bankroll | 1% |

The bankroll is queried live from Polymarket's CLOB API before every execution run, so Kelly sizing scales dynamically -- wins grow bets, losses shrink them.

---

## Ensemble Model

An optional Layer 4 that blends the base model with market prices using online logistic regression:

### Seven Features

```
x1: Net Rating Spread (home - away)
x2: Player Impact Sum (injury adjustments)
x3: Rest Differential
x4: Home Court Indicator (1.0)
x5: Pace Differential
x6: Market Price (Polymarket implied probability)
x7: CLV Residual (rolling 5-game model accuracy trend)
```

### Heuristic Weights (Before Training)

```
P = 0.45 * Model_Prob + 0.55 * Market_Price
```

Market price gets more weight because prediction markets are generally efficient -- the model's value comes from the marginal edges where markets are wrong, not from replacing market consensus.

**CLV-based dynamic reweighting:** When the model's recent CLV is positive (beating closing lines), the blend shifts more weight to the model:

```
If CLV_Residual > 0:
    Shift = Min(CLV_Residual * 2, 0.05)
    P = (0.45 + Shift) * Model + (0.55 - Shift) * Market
```

### Online Learning

The ensemble retrains via L2-regularized logistic regression:

```
Loss = BinaryCrossEntropy + 0.5 * ||w||^2
Learning rate: 0.01, Epochs: 500
```

- Retrains every 25 new games
- Uses expanding window up to 150 games (older games dropped for staleness)
- Minimum 45 games before switching from heuristic to learned weights
- Market price consistently emerges as the strongest feature

---

## Safety & Risk Management

### Circuit Breakers

**Brier score breaker:** Trips if any single day's Brier exceeds 0.35. This catches model failures before they compound.

**CLV breaker:** Trips if rolling 7-day CLV averages below -50 basis points on 10+ games. Negative CLV means the model is consistently buying at prices worse than where the market closes -- the model is adding negative information.

Both breakers require manual reset after investigation.

### Sanity Engine

Per-prediction validation:
- Probability bounds: 5% <= P <= 95%
- Impact magnitude: |info_edge| <= 15%
- Market divergence: |model - market| <= 20%
- Net rating sanity: |spread| <= 25 points

Daily batch checks:
- Maximum 25 games per day
- Favorite rate <= 45% (prevents systematic bias)
- Average confidence between 32-68%

### Trade Execution Safety

- **Stale signal guard:** Rejects trades if live price moved >1.5% from signal time
- **Minimum liquidity:** $500 in the orderbook
- **Daily loss limit:** Stops trading if daily P&L drops below -10% of bankroll
- **Max concurrent positions:** 10
- **Deduplication:** Prevents double-trading the same game

### Lineage Logging

Every prediction is logged to JSONL and Supabase with full decomposition: all layer inputs/outputs, market data at decision time, sanity status, model version. Complete audit trail for post-hoc analysis.

### Shadow Testing

When model changes are deployed, the old model runs in parallel. Alerts fire if:
- Max divergence between old and new exceeds 15%
- Alert rate (diffs > 5%) exceeds 20% of games

---

## Data Sources

| Source | Data | Sport | Auth Required |
|--------|------|-------|---------------|
| ESPN API | Scoreboard, injuries, rosters, results | NBA | No |
| NBA.com Stats | ORtg, DRtg, NetRtg, Pace per 100 | NBA | No (custom headers) |
| BallDontLie API | Historical player/team stats | NBA | API key |
| NHL API | Schedule, scores, standings, player stats | NHL | No |
| Daily Faceoff | Confirmed goalie starts | NHL | No (scrape) |
| Natural Stat Trick | xGF%, Corsi, Fenwick | NHL | No (scrape) |
| Polymarket Gamma API | Market discovery, event IDs | Both | No |
| Polymarket CLOB API | Live orderbook, trade execution | Both | Wallet signature |

**Cache policy:** Team stats cached 24h, injuries 30min, market prices 5min. Goalie starts and lineups are **never cached** -- always live.

---

## Infrastructure

### Database (Supabase PostgreSQL)

Core tables:
- **`games`** -- Daily schedule with triage levels, game times, results
- **`research`** -- Model outputs: probability decomposition, edge type (A1/A2/B/C), recommendation (BET/PASS)
- **`trades`** -- Executed positions: entry price, shares, outcome, P&L
- **`calibration`** -- Daily Brier scores, accuracy rates, attribution
- **`clv`** -- Entry price vs closing price for every traded game
- **`ensemble_training`** -- Feature snapshots for online model retraining
- **`prediction_lineage`** -- Full audit trail per prediction

### VPS (Hetzner CX23, Helsinki)

- Ubuntu 24.04
- Connected via Tailscale private network (public SSH blocked by UFW)
- Fresh Polymarket wallet: EOA + Gnosis Safe proxy
- Polygon RPC for on-chain redemptions

### Discord

Three bots posting to dedicated channels:
- `#daily-schedule` -- Morning slate with triage flags
- `#research-nba` / `#research-nhl` -- Per-game analysis threads
- `#trade-signals` -- Executed trades with fill details
- `#performance` -- Nightly grading reports
- `#alerts-nba` / `#alerts-nhl` -- Pre-game briefs

### Cron Schedule (All Times CT)

**Local:**
| Time | Job |
|------|-----|
| 7:00 AM | Morning slate |
| 7:30 AM | Research agent |
| Sunday 8:00 AM | Weekly recalibration |
| 8:00 AM | Alert agent daemon (checks every 5 min) |

**VPS:**
| Time | Job |
|------|-----|
| 1:30 AM (6:30 UTC) | Settler -- grade trades |
| 8:00 AM (13:00 UTC) | Redeem winning positions |
| 9:00 AM (14:00 UTC) | Schedule executor runs (at jobs based on game times) |
| Every 2 min (game hours) | CLV capture |
| Every 2 min (game hours) | Lineup monitor |

---

## Results & Findings

### Prediction Accuracy (14-day live window)

| Metric | Value |
|--------|-------|
| Games graded | 217 |
| Overall Brier score | **0.2078** |
| Best single day | 0.1386 (12 games) |
| Worst single day | 0.2517 (14 games) |
| NBA Brier | ~0.176 |
| NHL Brier | ~0.240 |

For context, a Brier score of 0.25 is equivalent to always predicting 50/50 (no skill). Lower is better. The model's 0.2078 represents meaningful predictive edge, especially in the NBA.

### Trading Performance (Trades #60--94)

| Metric | Value |
|--------|-------|
| Trades settled | 31 |
| Record | 18W-13L (58.1%) |
| Win/loss size ratio | 1.40x (avg winner larger than avg loser) |

The system was net profitable despite a modest win rate because winners were larger than losers (positive expected value via Kelly sizing).

### Key Findings

1. **NBA net ratings were the strongest single predictor.** Switching from Elo to ORtg/DRtg per 100 possessions improved NBA Brier by 0.015 (significant over hundreds of games).

2. **NHL is harder to predict.** NHL Brier (0.240) was consistently worse than NBA (0.176). Hockey's higher variance (lower scoring, goalie dependence, puck luck) makes it fundamentally less predictable. The system compensated with higher edge thresholds for NHL trades.

3. **Situational adjustments helped marginally.** Layer 2 improved Brier by ~0.0015 on average. Back-to-backs and rest differentials were the most reliable signals. Form and H2H were noisier.

4. **Injury impact had diminishing returns at Tier 3+.** Quality starters being out mattered, but stacking multiple Tier 3/4 injuries didn't compound linearly. The diminishing returns model (100%/50%/25%) was validated by backtest.

5. **Market prices are efficient but not perfect.** The ensemble model confirmed that Polymarket prices were the strongest single feature. The model's edge came from the ~5-15% of games where situational/information factors hadn't been priced in yet.

6. **CLV was negative overall (-40 to -84 bps).** The model was buying at prices slightly worse than closing lines on average. This suggests the market was incorporating the same information the model used, just slightly faster. Despite negative CLV, trades were still profitable because the model's directional accuracy was good enough.

7. **Quarter Kelly kept drawdowns manageable.** Even with a 1W-3L losing day, the bankroll recovered. Full Kelly would have risked ruin on that kind of streak.

### What Broke

- **Polymarket geoblock**: Local machine trade execution started failing. VPS handled it, but 130 trade attempts from the local research agent were wasted.
- **Snapshot serialization bug**: A `GameSnapshot.get()` AttributeError caused 0 games analyzed for the final 3 days. The model still computed edges (two-sided edge logs show it working) but couldn't persist results.
- **Bankroll depletion**: The trading wallet balance eventually became too small for meaningful positions.

---

## Project Structure

```
sports_polymarket/
├── src/
│   ├── model/
│   │   ├── baseline.py          # Elo + Net Rating models
│   │   ├── decomposition.py     # 3-layer probability combination
│   │   ├── edge.py              # Edge calculation + Kelly criterion
│   │   ├── situational.py       # B2B, rest, travel, form, H2H
│   │   ├── information.py       # Injury impact tiers
│   │   ├── nba_ratings.py       # NBA.com ORtg/DRtg integration
│   │   ├── ensemble.py          # Online logistic regression blend
│   │   ├── clv.py               # Closing line value tracking
│   │   └── safety.py            # Circuit breakers, sanity, lineage
│   ├── agents/
│   │   ├── research_agent.py    # Main analysis pipeline (~1600 lines)
│   │   ├── alert_agent.py       # Pre-game brief daemon
│   │   ├── performance_agent.py # Overnight grading + lessons
│   │   └── model_router.py      # LLM model selection
│   ├── data/
│   │   ├── nba_espn.py          # ESPN scoreboard + injuries
│   │   ├── nba_stats.py         # NBA.com advanced stats
│   │   ├── nhl_api.py           # Official NHL API
│   │   ├── nhl_goalies.py       # Goalie stats + starter detection
│   │   ├── nhl_dailyfaceoff.py  # Confirmed goalie starts
│   │   ├── nhl_naturalstattrick.py  # xGF%, Corsi, Fenwick
│   │   ├── polymarket.py        # Gamma + CLOB API integration
│   │   ├── h2h.py               # Head-to-head history
│   │   └── cache.py             # TTL-based caching layer
│   ├── trading/
│   │   ├── executor.py          # Trade execution + validation
│   │   └── settler.py           # Win/loss grading + P&L
│   └── scripts/
│       └── morning_slate.py     # Daily schedule generation
├── scripts/
│   ├── vps_executor.py          # VPS-side trade execution
│   ├── clv_capture.py           # Closing price capture at tip-off
│   ├── redeem_positions.py      # Gnosis Safe batch redemption
│   ├── schedule_executor.py     # Smart scheduling via `at` jobs
│   ├── recalibrate.py           # Weekly shrinkage re-optimization
│   ├── lineup_monitor.py        # Pre-game lineup change detection
│   ├── log_bet.py               # Manual bet logging CLI
│   └── backtest_runner.py       # Historical backtest harness
├── agents/                      # Agent identity/config files
├── migrations/                  # Supabase SQL schema (001-008)
├── backtest_results/            # Historical backtest CSVs
├── tests/                       # Unit + integration tests
└── logs/                        # Local + VPS log archive
    └── vps/                     # Archived VPS logs
```

---

## Setup

### Prerequisites

- Python 3.12+
- Supabase project (PostgreSQL)
- Polymarket wallet (EOA + approved allowances)
- Non-US VPS for trade execution (e.g. Hetzner)
- Discord server + bot tokens

### Environment Variables

```bash
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...

# Polymarket (VPS only)
POLY_PRIVATE_KEY=...
POLY_FUNDER_ADDRESS=...

# APIs
BALLDONTLIE_API_KEY=...
OPENROUTER_API_KEY=...

# Blockchain
POLYGON_RPC=https://polygon-bor-rpc.publicnode.com
```

### Database Setup

Run migrations in order:

```bash
psql $DATABASE_URL -f migrations/001_create_tables.sql
psql $DATABASE_URL -f migrations/002_backtest_table.sql
# ... through 008
```

### Running

```bash
# Morning slate
python3 src/scripts/morning_slate.py

# Research agent (full analysis)
python3 -m src.agents.research_agent

# Execute trades (VPS only)
python3 scripts/vps_executor.py          # live
python3 scripts/vps_executor.py --dry-run # preview

# Backtest
python3 scripts/backtest_runner.py
```

---

## License

This project is provided as-is for educational purposes. Sports betting involves financial risk. Past performance does not guarantee future results.
