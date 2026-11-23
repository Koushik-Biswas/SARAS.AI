<#
.SYNOPSIS
  SARAS AI development mode: FastAPI with auto-reload on :8000 and the Vite dev
  server on :5173 (which proxies /api to :8000), in one console.

.DESCRIPTION
  Windows PowerShell 5.1 compatible. Run from anywhere:
      powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1
      powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1 -ApiOnly
      powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1 -WebOnly

  Open http://localhost:5173. Press Ctrl+C to stop; the script then stops the API
  process tree it started (uvicorn's reloader spawns a child process).

  Prefer two terminals? Run these instead:
      cd backend;  .venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
      cd frontend; npm run dev

  On the 4 GB laptop, close other programs first and stop both servers when done.
#>
[CmdletBinding()]
param(
    [switch]$ApiOnly,
    [switch]$WebOnly
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'
$Python = Join-Path $Backend '.venv\Scripts\python.exe'
$ViteJs = Join-Path $Frontend 'node_modules\vite\bin\vite.js'

function Test-PortBusy([int]$Port) {
    $busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return [bool]$busy
}

function Stop-Tree([System.Diagnostics.Process]$Proc) {
    if ($null -ne $Proc -and -not $Proc.HasExited) {
        Write-Host '[saras] Stopping the API...' -ForegroundColor Green
        & taskkill.exe /PID $Proc.Id /T /F | Out-Null
    }
}

if (-not $WebOnly) {
    if (-not (Test-Path $Python)) {
        throw "Python virtual environment not found at $Python. Create it first: cd backend; py -3.12 -m venv .venv; .venv\Scripts\python -m pip install -r requirements-dev.txt"
    }
    if (Test-PortBusy 8000) { throw 'Port 8000 is already in use. Stop the other server first.' }
}
if (-not $ApiOnly) {
    if (Test-PortBusy 5173) { throw 'Port 5173 is already in use. Stop the other dev server first.' }
    if (-not (Test-Path $ViteJs)) {
        Write-Host '[saras] Installing the web app dependencies (first run)...' -ForegroundColor Green
        Push-Location $Frontend
        try { & npm.cmd install --no-audit --no-fund } finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) { throw 'npm install failed.' }
    }
}

$api = $null
try {
    if (-not $WebOnly) {
        Write-Host '[saras] API: http://127.0.0.1:8000  (docs: /api/docs)' -ForegroundColor Cyan
        $apiArgs = @('-m', 'uvicorn', 'app.main:app', '--reload', '--host', '127.0.0.1', '--port', '8000')
        if ($ApiOnly) {
            Push-Location $Backend
            try { & $Python @apiArgs } finally { Pop-Location }
            return
        }
        $api = Start-Process -FilePath $Python -ArgumentList $apiArgs -WorkingDirectory $Backend `
            -NoNewWindow -PassThru
    }
    Write-Host '[saras] Web app: http://localhost:5173' -ForegroundColor Magenta
    Push-Location $Frontend
    try {
        # node + vite.js directly (not npm.cmd) so Ctrl+C does not ask "Terminate batch job?"
        & node $ViteJs
    } finally {
        Pop-Location
    }
} finally {
    Stop-Tree $api
}
