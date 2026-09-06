@echo off
REM Launch the hybrid-rag terminal UI.
REM
REM Double-clicking this, or running it from PowerShell or cmd.exe, gets you a
REM real Windows console - which is what prompt_toolkit needs. Git Bash's MinTTY
REM does not provide one and the UI will tell you so rather than crash.
REM
REM `uv run` resolves the project's pinned 3.12 toolchain. The system
REM interpreter on this machine is 3.14, which this project does not support.

setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv is not on PATH. Falling back to the project virtual environment.
    if not exist ".venv\Scripts\hybrid-rag.exe" (
        echo No .venv found either. Run: uv sync --extra dev
        exit /b 1
    )
    ".venv\Scripts\hybrid-rag.exe" %*
    exit /b %errorlevel%
)

uv run hybrid-rag %*
exit /b %errorlevel%
