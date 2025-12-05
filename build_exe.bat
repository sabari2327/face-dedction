@echo off
REM Build standalone .exe file
REM This creates FaceRecognition.exe that runs without Python installed

echo.
echo ========================================
echo Building Standalone .EXE File
echo ========================================
echo.
echo This will take 2-3 minutes...
echo.

setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0

REM Clean old builds
if exist "%PROJECT_DIR%build" rmdir /s /q "%PROJECT_DIR%build"
if exist "%PROJECT_DIR%dist" rmdir /s /q "%PROJECT_DIR%dist"

REM Run PyInstaller
cd /d "%PROJECT_DIR%"
python -m PyInstaller FaceRecognition.spec --distpath "dist" --workpath "build" --noconfirm

if errorlevel 1 (
    echo.
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Your executable is ready at:
echo   %PROJECT_DIR%dist\FaceRecognition.exe
echo.
echo You can now:
echo 1. Run it on any Windows PC (no Python needed)
echo 2. Share dist\ folder with others
echo 3. Create a shortcut to FaceRecognition.exe
echo.
pause
