@echo off
REM Launch interactive face matcher: Upload image → Match in live camera

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
echo FACE MATCHER - Upload Image + Live Cam
echo ========================================
echo.
echo This tool lets you:
echo 1. Upload an image of a person
echo 2. See the face detected in that image
echo 3. Open live camera to find matching faces
echo.
echo Press Ctrl+C in the camera window to quit.
echo.
"%PYTHON_CMD%" match_faces_live.py

if errorlevel 1 (
    echo.
    echo ERROR: Script failed
    echo Make sure you've run:
    echo   - python extract_embeddings.py
    echo   - python train_model.py
    echo.
)

pause
