# Removes the daily retraining task from the Windows Task Scheduler.
#   powershell -ExecutionPolicy Bypass -File scripts\unregister_daily_retraining.ps1

param([string]$TaskName = "ETF Classifier - daily retraining")

$ErrorActionPreference = "Stop"
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Task '$TaskName' removed."
} else {
    Write-Host "No task named '$TaskName'."
}
