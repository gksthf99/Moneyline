#!/bin/bash
# Setup systemd service for the alert daemon.
# Run with sudo: sudo bash scripts/setup_systemd.sh

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cp "$PROJECT_DIR/scripts/sports-alert.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable sports-alert.service
systemctl start sports-alert.service
systemctl status sports-alert.service
