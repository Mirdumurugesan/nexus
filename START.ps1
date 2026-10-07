# ── NEXUS Startup Script ────────────────────────────────────────────────────
# Run this in PowerShell from the repo root:
#   .\START.ps1

Write-Host ""
Write-Host "  ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗" -ForegroundColor Cyan
Write-Host "  ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝" -ForegroundColor Cyan
Write-Host "  ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗" -ForegroundColor Cyan
Write-Host "  ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║" -ForegroundColor Cyan
Write-Host "  ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║" -ForegroundColor Cyan
Write-Host "  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Multi-Agent Autonomous Software Engineering Platform" -ForegroundColor White
Write-Host "  Planner → Engineer → Patch Gate + Reviewer ⇄ Reflector" -ForegroundColor DarkGray
Write-Host ""

# Activate venv
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    . .\venv\Scripts\Activate.ps1
    Write-Host "[✓] Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "[✗] Virtual environment not found. Run: python -m venv venv" -ForegroundColor Red
    exit 1
}

# Install/update dependencies
Write-Host "[→] Checking dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt -q
Write-Host "[✓] Dependencies ready" -ForegroundColor Green

Write-Host ""
Write-Host "[→] Starting NEXUS server..." -ForegroundColor Yellow
Write-Host "[→] Dashboard: http://127.0.0.1:8000/  (login at /login.html)" -ForegroundColor Cyan
Write-Host "[→] API Docs: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "[→] Press Ctrl+C to stop" -ForegroundColor DarkGray
Write-Host ""

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
