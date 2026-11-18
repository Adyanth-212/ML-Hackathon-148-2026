#!/usr/bin/env bash
# Laptop 1: start the API on all interfaces so Laptop 2 can reach it.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATHSNAP_MODE=serve
echo "Laptop 1 IP addresses (give one of these to the UI):"
(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null || true)
exec .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
