@echo off
SET EnvCheckLock=env_check.lock

if not exist %EnvCheckLock% (
	python scripts/check_requirements.py requirements.txt
	if errorlevel 1 (
    	echo Installing missing packages...
    	pip install -r requirements.txt
	)
	echo lock > %EnvCheckLock%
)
python -m autogpt %*
pause