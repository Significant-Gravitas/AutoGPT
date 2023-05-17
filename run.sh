#!/bin/bash

function find_python_command() {
    if command -v python &> /dev/null
    then
        echo "python"
    elif command -v python3 &> /dev/null
    then
        echo "python3"
    else
        echo "Python not found. Please install Python."
        exit 1
    fi
}

PYTHON_CMD=$(find_python_command)

if $PYTHON_CMD -c "import sys; sys.exit(sys.version_info < (3, 10))"; then
    $PYTHON_CMD scripts/check_requirements.py requirements.txt
    if [ $? -eq 1 ]
    then
        echo Installing missing packages...
        $PYTHON_CMD -m pip install -r requirements.txt
    fi
    $PYTHON_CMD -m autogpt $@
    read -p "Press any key to continue..."
else
    echo "Python 3.10 or higher is required to run Auto GPT."
fi