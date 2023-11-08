@echo off
setlocal enabledelayedexpansion

:FindPythonCommand
for %%A in (python python3) do (
    where /Q %%A
    if !errorlevel! EQU 0 (
        set "PYTHON_CMD=%%A"
        goto :Found
    )
)

echo Python not found. Please install Python.
pause
exit /B 1

:Found
%PYTHON_CMD% scripts/check_requirements.py
if errorlevel 1 (
    echo
    %PYTHON_CMD% -m poetry install --without dev
    echo
    echo Finished installing packages! Starting AutoGPT...
    echo
)
%PYTHON_CMD% -m autogpt %*
pause
