#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/.discord_config.json"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: $CONFIG_FILE not found"
  exit 1
fi

tmp_file="$(mktemp)"
cleanup() {
  rm -f "$tmp_file"
}
trap cleanup EXIT

echo "============================================"
echo "  Discord Sport Channel Configuration"
echo "============================================"
echo ""
echo "This updates sport-specific channel IDs and webhook URLs in:"
echo "  $CONFIG_FILE"
echo ""

prompt_required() {
  local prompt="$1"
  local value
  while true; do
    read -r -p "$prompt" value
    if [ -n "$value" ]; then
      printf '%s' "$value"
      return
    fi
    echo "Value is required."
  done
}

CH_ALERTS_NBA="$(prompt_required "#alerts-nba channel ID: ")"
echo ""
CH_ALERTS_NHL="$(prompt_required "#alerts-nhl channel ID: ")"
echo ""
CH_TRADE_SIGNALS_NBA="$(prompt_required "#trade-signals-nba channel ID: ")"
echo ""
CH_TRADE_SIGNALS_NHL="$(prompt_required "#trade-signals-nhl channel ID: ")"
echo ""
WEBHOOK_ALERTS_NBA="$(prompt_required "#alerts-nba webhook URL: ")"
echo ""
WEBHOOK_ALERTS_NHL="$(prompt_required "#alerts-nhl webhook URL: ")"
echo ""
WEBHOOK_TRADE_SIGNALS_NBA="$(prompt_required "#trade-signals-nba webhook URL: ")"
echo ""
WEBHOOK_TRADE_SIGNALS_NHL="$(prompt_required "#trade-signals-nhl webhook URL: ")"
echo ""

python3 - "$CONFIG_FILE" "$tmp_file" \
  "$CH_ALERTS_NBA" "$CH_ALERTS_NHL" \
  "$CH_TRADE_SIGNALS_NBA" "$CH_TRADE_SIGNALS_NHL" \
  "$WEBHOOK_ALERTS_NBA" "$WEBHOOK_ALERTS_NHL" \
  "$WEBHOOK_TRADE_SIGNALS_NBA" "$WEBHOOK_TRADE_SIGNALS_NHL" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
tmp_path = Path(sys.argv[2])
(
    ch_alerts_nba,
    ch_alerts_nhl,
    ch_trade_nba,
    ch_trade_nhl,
    wh_alerts_nba,
    wh_alerts_nhl,
    wh_trade_nba,
    wh_trade_nhl,
) = sys.argv[3:11]

config = json.loads(config_path.read_text())
channels = config.setdefault("channels", {})
webhooks = config.setdefault("webhooks", {})

channels["alerts_nba"] = ch_alerts_nba
channels["alerts_nhl"] = ch_alerts_nhl
channels["trade_signals_nba"] = ch_trade_nba
channels["trade_signals_nhl"] = ch_trade_nhl

webhooks["alerts_nba"] = wh_alerts_nba
webhooks["alerts_nhl"] = wh_alerts_nhl
webhooks["trade_signals_nba"] = wh_trade_nba
webhooks["trade_signals_nhl"] = wh_trade_nhl

tmp_path.write_text(json.dumps(config, indent=2) + "\n")
PY

mv "$tmp_file" "$CONFIG_FILE"

echo "Saved sport-specific Discord channel and webhook config."
echo "Updated: $CONFIG_FILE"
