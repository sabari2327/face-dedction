@echo off
rem One-click launcher to run live webcam recognition
rem Usage: double-click this file or run from PowerShell/CMD

setlocal
set PROJECT_DIR=%~dp0
if exist "%PROJECT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo Using %PYTHON%
cd /d "%PROJECT_DIR%"

rem Start live webcam preview (press q to quit in the preview window)
echo Starting live webcam preview. Press q to quit the preview window.
"%PYTHON%" recognize_video.py

endlocal
exit /b 0
