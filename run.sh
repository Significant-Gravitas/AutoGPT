#!/bin/bash
python3 scripts/check_requirements.py requirements.txt
if [ $? -eq 1 ]
then
    echo Installing missing packages...
    pip3 install -r requirements.txt
fi
python3 -m autogpt $@
read -p "Press any key to continue..."
