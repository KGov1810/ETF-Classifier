# Runs the daily retraining once (this is what the Windows Task Scheduler calls).
# The project folder is the parent of this script's folder; the virtual environment is .venv.
# Output is appended to logs\retraining_console.log; the exit code is the job's (1 = failure).

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project

$python = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found: $python (create it with: python -m venv .venv)"
    exit 1
}

$logs = Join-Path $project "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$console = Join-Path $logs "retraining_console.log"

"===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" | Out-File -FilePath $console -Append -Encoding utf8
# Windows PowerShell 5.1 turns every stderr line of a native program into an error record:
# with "Stop", the first warning printed by Python would abort the script. The exit code decides instead.
$ErrorActionPreference = "Continue"
& $python -m etf_classifier.retraining.run @args *>&1 | Out-File -FilePath $console -Append -Encoding utf8
exit $LASTEXITCODE
