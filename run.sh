#!/bin/bash
if [ ! -d "./env_check.lock" ];then
    python scripts/check_requirements.py requirements.txt
    if [ $? -eq 1 ]
    then
    echo Installing missing packages...
    pip install -r requirements.txt
    fi
    touch env_check.lock
fi
python -m autogpt $@
read -p "Press any key to continue..."
