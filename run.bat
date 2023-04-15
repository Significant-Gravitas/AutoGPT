@echo off
python scripts/check_requirements.py requirements.txt
if errorlevel 1 (
    echo Installing missing packages...
    pip install -r requirements.txt
)
python scripts/main.py %1
pause
