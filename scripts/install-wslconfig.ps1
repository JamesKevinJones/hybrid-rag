# One line. Installs the WSL2 memory caps and restarts the VM cleanly.
#
#   powershell -File scripts\install-wslconfig.ps1
#
# CAUTION: machine-wide, and `wsl --shutdown` stops every running distro and
# every container inside them. Read scripts\wslconfig.template first - the
# 6 GB ceiling is shared with Docker-in-WSL on this host.
Copy-Item "$PSScriptRoot\wslconfig.template" "$HOME\.wslconfig" -Force; wsl --shutdown; Write-Host "Installed $HOME\.wslconfig and shut WSL down. It restarts on next use." -ForegroundColor Green
