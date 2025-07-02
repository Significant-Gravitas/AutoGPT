@echo off
setlocal enabledelayedexpansion

goto :main

REM --- Helper: Check command existence ---
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

REM --- Variables ---
set SCRIPT_DIR=%~dp0
set LOG_DIR=%SCRIPT_DIR%logs
set BACKEND_LOG=%LOG_DIR%\backend_setup.log
set FRONTEND_LOG=%LOG_DIR%\frontend_setup.log
set CLONE_NEEDED=0
set REPO_DIR=%SCRIPT_DIR%..\..

REM --- Create logs folder immediately ---
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo Checking prerequisites...
call :check_command git Git
call :check_command docker Docker
call :check_command npm Node.js
call :check_command pnpm pnpm
echo.

REM --- Detect repo ---
if exist "%REPO_DIR%\.git" (
    set CLONE_NEEDED=0
) else (
    set REPO_DIR=%SCRIPT_DIR%AutoGPT
    set CLONE_NEEDED=1
)

REM --- Clone repo if needed ---
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

REM --- Prompt for Sentry enablement ---
set SENTRY_ENABLED=0
echo Would you like to enable debug information to be shared so we can fix your issues? [Y/n]
set /p sentry_answer="Enable Sentry? [Y/n]: "
if /I "%sentry_answer%"=="" set SENTRY_ENABLED=1
if /I "%sentry_answer%"=="y" set SENTRY_ENABLED=1
if /I "%sentry_answer%"=="yes" set SENTRY_ENABLED=1
if /I "%sentry_answer%"=="n" set SENTRY_ENABLED=0
if /I "%sentry_answer%"=="no" set SENTRY_ENABLED=0

REM --- Setup backend ---
echo Setting up backend services...
echo.
cd /d "%REPO_DIR%\autogpt_platform"
if exist .env.example copy /Y .env.example .env >nul
cd backend
if exist .env.example copy /Y .env.example .env >nul

REM --- Set SENTRY_DSN in backend/.env ---
set SENTRY_DSN=https://11d0640fef35640e0eb9f022eb7d7626@o4505260022104064.ingest.us.sentry.io/4507890252447744
if %SENTRY_ENABLED%==1 (
    powershell -Command "(Get-Content .env) -replace '^SENTRY_DSN=.*', 'SENTRY_DSN=%SENTRY_DSN%' | Set-Content .env"
    echo Sentry enabled in backend.
) else (
    powershell -Command "(Get-Content .env) -replace '^SENTRY_DSN=.*', 'SENTRY_DSN=' | Set-Content .env"
    echo Sentry not enabled in backend.
)
cd ..

docker compose down > "%BACKEND_LOG%" 2>&1
if errorlevel 1 echo (docker compose down failed, continuing...)
docker compose up -d --build >> "%BACKEND_LOG%" 2>&1
if errorlevel 1 (
    echo Backend setup failed. See log: %BACKEND_LOG%
    pause
    exit /b 1
)
echo Backend services started successfully.
echo.

REM --- Setup frontend ---
echo Setting up frontend application...
echo.
cd frontend
if exist .env.example copy /Y .env.example .env >nul
call pnpm.cmd install
if errorlevel 1 (
    echo pnpm install failed!
    pause
    exit /b 1
)
echo Frontend dependencies installed successfully.
echo.

REM --- Start frontend dev server in the same terminal ---
echo Setup complete!
echo Access AutoGPT at: http://localhost:3000
echo To stop services, press Ctrl+C and run "docker compose down" in %REPO_DIR%\autogpt_platform
echo.
echo The frontend will now start in this terminal. Closing this window will stop the frontend.
echo Press Ctrl+C to stop the frontend at any time.
echo.

call pnpm.cmd dev