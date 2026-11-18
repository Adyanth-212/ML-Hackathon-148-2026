#!/usr/bin/env bash
# Laptop 1: start the API on all interfaces so Laptop 2 can reach it.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATHSNAP_MODE=serve
printf 'PathSnap API · laptop addresses:\n'
(hostname -I 2>/dev/null || true)
exec .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
