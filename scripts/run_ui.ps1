# Laptop 2 (Windows): set the API laptop's LAN address before running.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not $env:PATHSNAP_API_URL) {
    throw 'Set PATHSNAP_API_URL, e.g. $env:PATHSNAP_API_URL = "http://192.168.1.20:8000"'
}
python -m streamlit run ui/app.py
