#!/bin/bash
cd /opt/sports_polymarket
set -a
source .env
set +a
source venv/bin/activate
python3 scripts/schedule_executor.py "$@"
