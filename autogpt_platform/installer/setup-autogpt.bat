@echo off
setlocal enabledelayedexpansion

REM Variables
set SCRIPT_DIR=%~dp0
set REPO_DIR=%SCRIPT_DIR%..\..
set CLONE_NEEDED=0
set LOG_FILE=

echo =============================
echo   AutoGPT Windows Setup
echo =============================
echo.

REM Check prerequisites
echo Checking prerequisites...
where git >nul 2>nul
if errorlevel 1 (
    echo Git is not installed. Please install it and try again.
    pause
    exit /b 1
)
echo Git is installed.

where docker >nul 2>nul
if errorlevel 1 (
    echo Docker is not installed. Please install it and try again.
    pause
    exit /b 1
)
echo Docker is installed.
echo.

REM Detect repo
if exist "%REPO_DIR%\.git" (
    echo Using existing AutoGPT repository.
    set CLONE_NEEDED=0
) else (
    set REPO_DIR=%SCRIPT_DIR%AutoGPT
    set CLONE_NEEDED=1
)

REM Clone repo if needed
if %CLONE_NEEDED%==1 (
    echo Cloning AutoGPT repository...
    git clone https://github.com/Significant-Gravitas/AutoGPT.git "%REPO_DIR%"
    if errorlevel 1 (
        echo Failed to clone repository.
        pause
        exit /b 1
    )
    echo Repository cloned successfully.
)
echo.

REM Navigate to autogpt_platform
cd /d "%REPO_DIR%\autogpt_platform"
if errorlevel 1 (
    echo Failed to navigate to autogpt_platform directory.
    pause
    exit /b 1
)

REM Create logs directory
if not exist logs mkdir logs

REM Run docker compose with logging
echo Starting AutoGPT services with Docker Compose...
echo This may take a few minutes on first run...
echo.
set LOG_FILE=%REPO_DIR%\autogpt_platform\logs\docker_setup.log
docker compose up -d > "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo Docker compose failed. Check log file for details: %LOG_FILE%
    echo.
    echo Common issues:
    echo - Docker is not running
    echo - Insufficient disk space  
    echo - Port conflicts (check if ports 3000, 8000, etc. are in use)
    pause
    exit /b 1
)

echo =============================
echo      Setup Complete!
echo =============================
echo.
echo Access AutoGPT at: http://localhost:3000
echo API available at: http://localhost:8000
echo.
echo To stop services: docker compose down
echo To view logs: docker compose logs -f
echo.
echo Press any key to exit (services will keep running)...
pause >nul