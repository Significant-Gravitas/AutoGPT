@echo off

setlocal

set PY_CMD=python3
where %PY_CMD% >nul 2>nul || set PY_CMD=python

%PY_CMD% scripts/check_requirements.py requirements.txt
if errorlevel 1 (
    echo Installing missing packages...
    pip install -r requirements.txt
)

%PY_CMD% -m autogpt %*
pause
