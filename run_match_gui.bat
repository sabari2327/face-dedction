@echo off
REM Launch interactive face matcher with GUI file picker
REM Upload image directly by clicking - no typing paths!

setlocal enabledelayedexpansion

REM Get project directory
set PROJECT_DIR=%~dp0

REM Check for virtual environment
if exist "%PROJECT_DIR%.venv\Scripts\python.exe" (
    set PYTHON_CMD=%PROJECT_DIR%.venv\Scripts\python.exe
) else (
    set PYTHON_CMD=python
)

REM Run the script
cd /d "%PROJECT_DIR%"
echo.
echo ========================================
echo FACE MATCHER - Pick Image + Live Cam
echo ========================================
echo.
echo A file picker will open to select an image.
echo Then enter the person's name and watch live camera matching!
echo.
echo Controls in camera:
echo   q - Quit
echo   s - Save snapshot
echo.
"%PYTHON_CMD%" match_faces_gui.py

if errorlevel 1 (
    echo.
    echo ERROR: Script failed
    echo Make sure you have all model files in the project folder
    echo.
)

pause
