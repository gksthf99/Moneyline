#!/bin/bash
# Setup Polymarket trading credentials in .env
# Usage: bash scripts/setup_polymarket.sh

set -e

ENV_FILE="$(dirname "$0")/../.env"

echo "=== Polymarket Trading Setup ==="
echo ""

# Private key
read -rp "Polygon wallet private key (with or without 0x prefix): " PRIVATE_KEY
# Strip 0x prefix if present
PRIVATE_KEY="${PRIVATE_KEY#0x}"

# Signature type
echo ""
echo "Signature types:"
echo "  0 = EOA (MetaMask / hardware wallet / raw private key)"
echo "  1 = Email / Magic wallet"
read -rp "Signature type [0]: " SIG_TYPE
SIG_TYPE="${SIG_TYPE:-0}"

# Funder address (only for sig type 1)
FUNDER=""
if [ "$SIG_TYPE" = "1" ]; then
    read -rp "Funder address: " FUNDER
fi

# Bankroll
echo ""
read -rp "Starting bankroll in USD [1000]: " BANKROLL
BANKROLL="${BANKROLL:-1000}"

# Append to .env
echo "" >> "$ENV_FILE"
echo "# Polymarket Trading" >> "$ENV_FILE"
echo "POLY_PRIVATE_KEY=$PRIVATE_KEY" >> "$ENV_FILE"
echo "POLY_SIGNATURE_TYPE=$SIG_TYPE" >> "$ENV_FILE"
echo "POLY_FUNDER_ADDRESS=$FUNDER" >> "$ENV_FILE"
echo "POLY_BANKROLL=$BANKROLL" >> "$ENV_FILE"

echo ""
echo "Written to $ENV_FILE"
echo ""
echo "Next steps:"
echo "  1. Fund your wallet with USDC on Polygon"
echo "  2. Approve token allowances: python3 scripts/approve_allowances.py"
echo "  3. Trading will auto-execute on next research agent run (7:30 AM cron)"
