<#
.SYNOPSIS
  SARAS AI low-resource profile (PRD 18.6, the 4 GB laptop): build the web app once,
  then run ONE uvicorn process that serves the API, the built SPA and the in-process
  job worker on http://127.0.0.1:8000.

.DESCRIPTION
  Windows PowerShell 5.1 compatible. Run from anywhere:
      powershell -ExecutionPolicy Bypass -File scripts\run_prod.ps1
      powershell -ExecutionPolicy Bypass -File scripts\run_prod.ps1 -SkipBuild
      powershell -ExecutionPolicy Bypass -File scripts\run_prod.ps1 -BindHost 0.0.0.0 -Port 8080

  No Docker, no Node at runtime, SQLite in backend\data\. Settings come from
  backend\.env (copy backend\.env.example). Set SARAS_JWT_SECRET there before
  letting anyone else use the server. Press Ctrl+C to stop.
#>
[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [string]$BindHost = '127.0.0.1',
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'
$Python = Join-Path $Backend '.venv\Scripts\python.exe'
$Index = Join-Path $Frontend 'dist\index.html'

if (-not (Test-Path $Python)) {
    throw "Python virtual environment not found at $Python. Create it first: cd backend; py -3.12 -m venv .venv; .venv\Scripts\python -m pip install -r requirements.txt"
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Stop the other server first."
}

# 1. Build the SPA (frontend\dist). Skipped with -SkipBuild when a build already exists.
if (-not $SkipBuild -or -not (Test-Path $Index)) {
    Push-Location $Frontend
    try {
        if (-not (Test-Path (Join-Path $Frontend 'node_modules'))) {
            Write-Host '[saras] Installing the web app dependencies...' -ForegroundColor Green
            & npm.cmd ci --no-audit --no-fund
            if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
        }
        Write-Host '[saras] Building the web app (frontend\dist)...' -ForegroundColor Green
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'The web app build failed.' }
    } finally {
        Pop-Location
    }
}

# 2. Warn when the development JWT secret would be used.
$envFile = Join-Path $Backend '.env'
$secretSet = [bool]$env:SARAS_JWT_SECRET
if (-not $secretSet -and (Test-Path $envFile)) {
    $secretSet = [bool](Select-String -Path $envFile -Pattern '^\s*SARAS_JWT_SECRET\s*=\s*\S' -Quiet)
}
if (-not $secretSet) {
    Write-Warning 'SARAS_JWT_SECRET is not set: using the insecure development secret. Set it in backend\.env before sharing this server.'
}

# 3. One process, no --reload, worker thread inside it.
Write-Host "[saras] SARAS AI on http://${BindHost}:$Port  (API docs: /api/docs). Ctrl+C to stop." -ForegroundColor Cyan
Push-Location $Backend
try {
    & $Python -m uvicorn app.main:app --host $BindHost --port $Port
} finally {
    Pop-Location
}
