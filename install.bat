@echo off
rem This BAT file installs the required packages from a requirements.txt file for AutoGPT.

rem Check if Python is installed
where python >nul 2>nul
if errorlevel 1 (
    echo Python is not installed or not in your PATH. Please install Python and try again.
    pause
    exit /b 1
)

rem Check if pip is installed
where pip >nul 2>nul
if errorlevel 1 (
    echo pip is not installed or not in your PATH. Please install pip and try again.
    pause
    exit /b 1
)

rem Check if requirements.txt exists
if not exist requirements.txt (
    echo requirements.txt not found. Please ensure it is in the same directory as this script.
    pause
    exit /b 1
)

rem Install packages from requirements.txt
echo Installing packages from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo An error occurred while installing packages. Please check the requirements.txt file and try again.
    pause
    exit /b 1
)

echo Installation completed successfully!
pause
