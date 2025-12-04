@echo off
rem One-click launcher to run full pipeline: extract embeddings -> train -> save annotated image
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

echo Running face embeddings extraction...
"%PYTHON%" extract_embeddings.py || goto :err

echo Training recognizer model...
"%PYTHON%" train_model.py || goto :err

echo Running recognition on sample image and saving annotated output...
"%PYTHON%" recognize_image.py --image images\openface.jpg --output output\result.jpg || goto :err

echo Done. Opening result image...
start "" "%PROJECT_DIR%output\result.jpg"
pause
endlocal
exit /b 0

:err
echo An error occurred. Check the console output for details.
pause
endlocal
exit /b 1
