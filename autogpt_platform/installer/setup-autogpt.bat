@echo off
setlocal enabledelayedexpansion

:: AutoGPT Setup Script for Windows

call :print_banner

echo AutoGPT's Automated Setup Script
echo -------------------------------
echo This script will automatically install and set up AutoGPT for you.
echo.
echo Checking prerequisites:

:: Check if git is installed
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Git is not installed. Please install Git and try again.
    echo Visit https://git-scm.com/downloads for installation instructions.
    echo Press Enter to exit...
    pause >nul
    exit /b 1
) else (
    echo - Git is installed
)

:: Check if docker is installed
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Docker is not installed. Please install Docker and try again.
    echo Visit https://docs.docker.com/get-docker/ for installation instructions.
    echo Press Enter to exit...
    pause >nul
    exit /b 1
) else (
    echo - Docker is installed
)

:: Check if docker is running
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Docker is not running. Please start Docker Desktop and try again.
    echo Press Enter to exit...
    pause >nul
    exit /b 1
)

:: Set Docker commands
set "DOCKER_CMD=docker"
set "DOCKER_COMPOSE_CMD=docker compose"

:: Check if npm is installed
where npm >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: npm is not installed. Please install Node.js and npm and try again.
    echo Visit https://nodejs.org/en/download/ for installation instructions.
    echo Press Enter to exit...
    pause >nul
    exit /b 1
) else (
    echo - npm is installed
)

echo All prerequisites are installed! Starting installation...
echo.

:: Check if AutoGPT directory already exists
if exist "AutoGPT" (
    echo AutoGPT directory already exists.
    choice /C YN /M "Do you want to remove it and clone again? (Y/N)"
    if !ERRORLEVEL! equ 1 (
        echo Removing existing AutoGPT directory...
        rmdir /S /Q AutoGPT
        if !ERRORLEVEL! neq 0 (
            echo Failed to remove existing directory. Please check permissions and try again.
            echo Press Enter to exit...
            pause >nul
            exit /b 1
        )
    ) else (
        echo Using existing AutoGPT directory...
        cd AutoGPT
        goto :setup_backend
    )
)

:: Clone the repository
echo Cloning the AutoGPT repository...
git clone https://github.com/Significant-Gravitas/AutoGPT.git
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Failed to clone the repository. Please check your internet connection and try again.
    echo Press Enter to exit...
    pause >nul
    exit /b 1
)

cd AutoGPT

:setup_backend
echo Running backend and frontend setup!

:: Setup backend
echo Setting up backend services...
cd autogpt_platform
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Failed to navigate to AutoGPT\autogpt_platform directory.
    echo Press Enter to exit...
    pause >nul
    exit /b 1
)

:: Copy the example environment file
copy .env.example .env
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Failed to copy environment file. Please check permissions and try again.
    echo Press Enter to exit...
    pause >nul
    exit /b 1
)

:: Run docker compose
echo Starting backend services with Docker...
%DOCKER_COMPOSE_CMD% up -d --build
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Failed to start the backend services. Please check Docker and try again.
    echo Press Enter to exit...
    pause >nul
    exit /b 1
)

echo Backend services started successfully!
cd ..

:: Setup frontend
echo Setting up frontend application...
cd autogpt_platform\frontend
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Failed to navigate to frontend directory.
    echo Press Enter to exit...
    pause >nul
    exit /b 1
)

:: Copy the frontend example environment file
copy .env.example .env
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Failed to copy frontend environment file. Please check permissions and try again.
    echo Press Enter to exit...
    pause >nul
    exit /b 1
)

:: Install dependencies
echo Installing frontend dependencies...
call npm install
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Failed to install frontend dependencies. Please check npm and try again.
    echo Press Enter to exit...
    pause >nul
    exit /b 1
)

echo Frontend dependencies installed successfully!

:: Start the frontend development server
echo Starting frontend development server...
start cmd /k "cd %CD% && npm run dev"

echo AutoGPT setup completed successfully!
echo -------------------------------------
echo Your backend services are running in Docker.
echo Your frontend application is running at http://localhost:3000
echo.
echo Visit http://localhost:3000 in your browser to access AutoGPT.
echo.
echo To stop the services, close the npm terminal window and run 'docker compose down' in the AutoGPT\autogpt_platform directory.
echo.
echo Press Enter to exit the script (this will NOT stop the services)...
pause >nul
exit /b 0

:print_banner
echo.
echo        d8888          888             d8888b.  8888888b. 88888888888 
echo       d88888          888            d88P  Y88b 888   Y88b    888     
echo      d88P888          888            888    888 888    888    888     
echo     d88P 888 888  888 888888 d88b.   888        888   d88P    888     
echo    d88P  888 888  888 888   d88""88b 888  88888 8888888P"     888     
echo   d88P   888 888  888 888   888  888 888    888 888           888     
echo  d8888888888 Y88b 888 Y88b. Y88..88P Y88b  d88P 888           888     
echo d88P     888  "Y88888  "Y888 "Y88P"   "Y8888P88 888           888     
echo.
exit /b 0 