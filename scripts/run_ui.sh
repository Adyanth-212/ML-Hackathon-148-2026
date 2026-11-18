#!/usr/bin/env bash
# Laptop 2: start the UI. Set PATHSNAP_API_URL to Laptop 1's address first.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PATHSNAP_API_URL:?Set PATHSNAP_API_URL, e.g. export PATHSNAP_API_URL=http://192.168.1.20:8000}"
exec streamlit run ui/app.py
