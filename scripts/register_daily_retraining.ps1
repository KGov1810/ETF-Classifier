# Creates (or updates) the daily retraining task in the Windows Task Scheduler, for the current user.
#   powershell -ExecutionPolicy Bypass -File scripts\register_daily_retraining.ps1 [-Time 10:00]
# The task runs only while your Windows session is open (no administrator rights, no stored password),
# starts as soon as possible if the scheduled time was missed, and never runs twice at the same time.

param(
    [string]$Time = "10:00",
    [string]$TaskName = "ETF Classifier - daily retraining"
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_daily_retraining.ps1"
$project = Split-Path -Parent $PSScriptRoot

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`"" -WorkingDirectory $project
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "Daily retraining of the ETF Classifier model (promoted only if not worse)." `
    -Force | Out-Null

Write-Host "Task '$TaskName' registered: every day at $Time (or as soon as possible after a missed start)."
Write-Host "Run it now to test:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Last result:         Get-ScheduledTaskInfo -TaskName '$TaskName'"
