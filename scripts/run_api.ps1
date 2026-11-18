# Laptop 1 (Windows): .\.venv\Scripts\Activate.ps1, then run this script.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$env:PATHSNAP_MODE = 'serve'
Write-Host 'Starting PathSnap API on http://0.0.0.0:8000'
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
