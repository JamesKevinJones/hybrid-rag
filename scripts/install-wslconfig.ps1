<#
.SYNOPSIS
    Install the WSL2 memory caps, or check that they are ready to install.

.DESCRIPTION
    -Check   Validates the template and reports what WOULD change. Touches
             nothing, starts nothing, stops nothing. Safe to run any time.

    (none)   Copies the template to $HOME\.wslconfig and runs `wsl --shutdown`.

.EXAMPLE
    powershell -File scripts\install-wslconfig.ps1 -Check
    powershell -File scripts\install-wslconfig.ps1

.NOTES
    CAUTION - the install path is machine-wide and disruptive. `wsl --shutdown`
    stops EVERY distro and every container running inside them, with no
    warning to whatever is using them. The 6 GB ceiling is shared with
    Docker-in-WSL on this host, so containers sized above it will be
    OOM-killed after the restart.

    That is why -Check exists and why nothing in this repo runs the install
    path automatically. Read scripts\wslconfig.template, decide the ceiling is
    right for your workload, then run it yourself when nothing is mid-flight.
#>

[CmdletBinding()]
param([switch]$Check)

$ErrorActionPreference = 'Stop'
$template = Join-Path $PSScriptRoot 'wslconfig.template'
$target = Join-Path $HOME '.wslconfig'

if (-not (Test-Path $template)) {
    Write-Error "Template missing: $template"
    exit 1
}

# Pull the caps out of the template so the report states what is actually in
# the file rather than what this script's author remembered.
$content = Get-Content $template -Raw
$memory = if ($content -match '(?m)^\s*memory\s*=\s*(\S+)') { $Matches[1] } else { '(not set)' }
$swap = if ($content -match '(?m)^\s*swap\s*=\s*(\S+)') { $Matches[1] } else { '(not set)' }
$processors = if ($content -match '(?m)^\s*processors\s*=\s*(\S+)') { $Matches[1] } else { '(not set)' }

if ($Check) {
    Write-Host "WSL2 config readiness check" -ForegroundColor Cyan
    Write-Host "  template   : $template"
    Write-Host "  memory cap : $memory"
    Write-Host "  swap cap   : $swap"
    Write-Host "  processors : $processors"
    Write-Host "  target     : $target"

    if (Test-Path $target) {
        $existing = Get-Content $target -Raw
        if ($existing -eq ($content)) {
            Write-Host "  status     : ALREADY INSTALLED and identical" -ForegroundColor Green
        } else {
            Write-Host "  status     : a DIFFERENT .wslconfig already exists" -ForegroundColor Yellow
            Write-Host "               installing would overwrite it; diff them first"
        }
    } else {
        Write-Host "  status     : not installed - template is ready to deploy" -ForegroundColor Green
    }

    try {
        # wsl.exe emits UTF-16LE; without this the distro names come back with a
        # space between every character.
        $previousEncoding = [Console]::OutputEncoding
        [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
        $distros = ((wsl --list --quiet 2>$null) | Where-Object { $_.Trim() }) -join ', '
        [Console]::OutputEncoding = $previousEncoding
        if ($distros) {
            Write-Host "  WSL distros: $distros"
            Write-Host "               `wsl --shutdown` would stop all of these and any"
            Write-Host "               containers inside them." -ForegroundColor Yellow
        } else {
            Write-Host "  WSL distros: none detected"
        }
    } catch {
        Write-Host "  WSL distros: could not query (WSL may not be installed)"
    }

    Write-Host ""
    Write-Host "No changes made. Re-run without -Check to install." -ForegroundColor Cyan
    exit 0
}

Copy-Item $template $target -Force
Write-Host "Installed $target" -ForegroundColor Green
Write-Host "Shutting WSL down so the new caps take effect on next use..." -ForegroundColor Yellow
wsl --shutdown
Write-Host "Done. WSL restarts automatically the next time it is used." -ForegroundColor Green
