@echo off
python scripts/check_requirements.py requirements.txt
python -m autogpt %*
pause
