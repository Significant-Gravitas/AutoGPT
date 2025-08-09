@echo off
setlocal enabledelayedexpansion

REM Variables
set SCRIPT_DIR=%~dp0
set LOG_DIR=%SCRIPT_DIR%logs
set REPO_DIR=%SCRIPT_DIR%..\..
set CLONE_NEEDED=0
set SENTRY_ENABLED=0
set LOG_FILE=

REM Helper: Check command existence
:check_command
if "%1"=="" (
    echo ERROR: check_command called with no command argument!
    pause
    exit /b 1
)
where %1 >nul 2>nul
if errorlevel 1 (
    echo %2 is not installed. Please install it and try again.
    pause
    exit /b 1
) else (
    echo %2 is installed.
)
goto :eof

:main
echo =============================
echo   AutoGPT Windows Setup
echo =============================
echo.

REM Create logs folder immediately
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo Checking prerequisites...
call :check_command git Git
call :check_command docker Docker
echo.

REM Detect repo
if exist "%REPO_DIR%\.git" (
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
) else (
    echo Using existing AutoGPT repository.
)
echo.

REM Prompt for Sentry enablement
set SENTRY_ENABLED=0
echo Enable debug info sharing to help fix issues? [Y/n]
set /p sentry_answer="Enable Sentry? [Y/n]: "
if /I "%sentry_answer%"=="" set SENTRY_ENABLED=1
if /I "%sentry_answer%"=="y" set SENTRY_ENABLED=1
if /I "%sentry_answer%"=="yes" set SENTRY_ENABLED=1
if /I "%sentry_answer%"=="n" set SENTRY_ENABLED=0
if /I "%sentry_answer%"=="no" set SENTRY_ENABLED=0
echo.

REM Navigate to autogpt_platform
echo Setting up environment...
cd /d "%REPO_DIR%\autogpt_platform"
if errorlevel 1 (
    echo Failed to navigate to autogpt_platform directory.
    pause
    exit /b 1
)

REM Copy main .env
if exist .env.example copy /Y .env.example .env >nul
if errorlevel 1 (
    echo Failed to copy main .env file.
    pause
    exit /b 1
)

REM Configure backend Sentry
cd backend
if errorlevel 1 (
    echo Failed to navigate to backend directory.
    pause
    exit /b 1
)

if exist .env.example copy /Y .env.example .env >nul
if errorlevel 1 (
    echo Failed to copy backend .env file.
    pause
    exit /b 1
)

set SENTRY_DSN=https://11d0640fef35640e0eb9f022eb7d7626@o4505260022104064.ingest.us.sentry.io/4507890252447744
if %SENTRY_ENABLED%==1 (
    powershell -Command "(Get-Content .env) -replace '^SENTRY_DSN=.*', 'SENTRY_DSN=%SENTRY_DSN%' | Set-Content .env"
    echo Sentry enabled
) else (
    powershell -Command "(Get-Content .env) -replace '^SENTRY_DSN=.*', 'SENTRY_DSN=' | Set-Content .env"
    echo Sentry disabled
)

cd ..
echo.

REM Run docker compose with logging
echo Running docker compose up -d --build...
set LOG_FILE=%REPO_DIR%\autogpt_platform\logs\docker_setup.log
docker compose up -d --build > "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo Docker compose failed. Check log file for details: %LOG_FILE%
    pause
    exit /b 1
)

echo Services started successfully.
echo.
echo Setup complete!
echo Access AutoGPT at: http://localhost:3000
echo To stop services, run "docker compose down" in %REPO_DIR%\autogpt_platform
echo.
echo Press any key to exit (services will keep running)...
pause >nul