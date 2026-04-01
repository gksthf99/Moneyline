#!/bin/bash
# Setup cron jobs for the sports prediction system.
# Run this once: bash scripts/setup_cron.sh

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/usr/bin/python3"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

# Build crontab entries
CRON_ENTRIES="
# Sports Polymarket Prediction System
# Morning Slate — 7:00 AM CDT daily
0 7 * * * cd $PROJECT_DIR && $PYTHON src/scripts/morning_slate.py >> $LOG_DIR/morning_slate.log 2>&1

# Performance Grading — 12:30 AM CDT daily (grades yesterday's games)
30 0 * * * cd $PROJECT_DIR && $PYTHON -m src.agents.performance_agent >> $LOG_DIR/performance.log 2>&1
"

# Add to crontab (preserve existing entries)
(crontab -l 2>/dev/null | grep -v "sports_polymarket"; echo "$CRON_ENTRIES") | crontab -

echo "Cron jobs installed:"
crontab -l | grep "sports_polymarket"
echo ""
echo "Log directory: $LOG_DIR"
echo ""
echo "Alert daemon must be set up separately via systemd (see below)."
