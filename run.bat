@echo off
setlocal enabledelayedexpansion

:FindPythonCommand
for %%A in (python python3) do (
    where /Q %%A
    if !errorlevel! EQU 0 (
        set "PYTHON_CMD=%%A" & goto :FoundPython
    )
)

echo Python not found. Please install Python.
pause
exit /B 1

:FoundPython
set "VENV_DIR=%~dp0.venv"
if not exist "%VENV_DIR%" (
    echo Virtual environment not found. Please create a virtual environment at %VENV_DIR%.
    pause
    exit /B 1
)

set "ACTIVATE="
for %%A in (%VENV_DIR%\Scripts\activate.bat %VENV_DIR%\Scripts\Activate.ps1) do (
    if exist "%%A" (
        set "ACTIVATE=%%A" & goto :ActivateFound
    )
)

echo activate.bat or Activate.ps1 not found. Please check the virtual environment directory.
pause
exit /B 1

:ActivateFound
call "%ACTIVATE%"

set "AUTORUN="
for %%A in (%VENV_DIR%\Scripts\autorun.bat %VENV_DIR%\Scripts\autorun.ps1) do (
    if exist "%%A" (
        set "AUTORUN=%%A"
        set /p "USE_AUTORUN=Do you want to use the autorun script at %AUTORUN%? [Y/n] "
        if /i "!USE_AUTORUN!"=="n" (
            set "AUTORUN="
        ) else (
            goto :AutotunFound
        )
    )
)

:AutotunFound

:CheckRequirements
rem Check if the required packages are installed
%PYTHON_CMD% scripts/check_requirements.py %~dp0requirements.txt

if errorlevel 1 (
    echo Installing missing packages ...
    rem Install missing packages using pip
    %PYTHON_CMD% -m pip install -r %~dp0requirements.txt
)

if defined AUTORUN (
    rem Call the autorun script if it exists
    if /i "!ACTIVATE:~-3!"=="ps1" (
        powershell.exe -ExecutionPolicy RemoteSigned -File "%AUTORUN%"
    ) else (
        call "%AUTORUN%"
    )
)

rem Run the autogpt command
%PYTHON_CMD% -m autogpt %*

:InstallPluginDeps
rem Install the plugin dependencies
echo Installing plugin dependencies ...
%PYTHON_CMD% scripts/install_plugin_deps.py

pause
