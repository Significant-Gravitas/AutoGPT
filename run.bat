@echo off
if not exist env_check.lock (
	python scripts/check_requirements.py requirements.txt
	if errorlevel 1 (
    	echo Installing missing packages...
    	pip install -r requirements.txt
	)
	echo lock => env_check.lock
)
python -m autogpt %*
pause
