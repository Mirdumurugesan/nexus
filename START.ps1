# ── NEXUS Startup Script ────────────────────────────────────────────────────
# Run this in PowerShell from F:\nexus:
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
Write-Host "  Phase 2-4: Planner → Engineer → Reviewer → Reflector" -ForegroundColor DarkGray
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
pip install langgraph==0.2.28 -q
pip install langchain-core -q --upgrade
Write-Host "[✓] Dependencies ready" -ForegroundColor Green

Write-Host ""
Write-Host "[→] Starting NEXUS server..." -ForegroundColor Yellow
Write-Host "[→] Dashboard: open frontend\index.html in your browser" -ForegroundColor Cyan
Write-Host "[→] API Docs: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "[→] Press Ctrl+C to stop" -ForegroundColor DarkGray
Write-Host ""

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
