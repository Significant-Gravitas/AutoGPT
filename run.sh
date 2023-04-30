#!/bin/bash
if python3 -c "import sys; sys.exit(sys.version_info < (3, 10))"; then
    python3 scripts/check_requirements.py requirements.txt
    if [ $? -eq 1 ]
    then
        echo Installing missing packages...
        pip3 install -r requirements.txt
    fi
    python3 -m autogpt $@
    read -p "Press any key to continue..."
else
    echo "Python 3.10 or higher is required to run Auto GPT."
fi
