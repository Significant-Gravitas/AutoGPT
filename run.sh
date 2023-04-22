#!/bin/bash
EnvCheckLock=env_check.lock
if [ ! -r $EnvCheckLock ];then
    python scripts/check_requirements.py requirements.txt
    if [ $? -eq 1 ]
    then
    echo Installing missing packages...
    pip install -r requirements.txt
    fi
    touch $EnvCheckLock
fi
python -m autogpt "$@"
read -p "Press any key to continue..."