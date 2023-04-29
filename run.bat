
@echo off
python scripts/check_requirements.py requirements.txt
if errorlevel 1 (
    echo Installing missing packages...
    pip install -r requirement%s.txt
)
python -m autogpt %*


pip install -r requirements.txt


python -m autogpt %

 pip install -r requirement%s.txt

 




 







