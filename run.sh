#!/bin/bash

# Check if python or python3 is installed
if command -v python >/dev/null 2>&1; then
    PY=python
    PIP=pip
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
    PIP=pip3
else
    echo "Python is not installed. Please install Python 2.7 or 3.x."
    exit 1
fi

# Check if the required packages are installed
$PY scripts/check_requirements.py requirements.txt
if [ $? -eq 1 ]
then
    echo Installing missing packages...
    $PIP install -r requirements.txt
fi

# Run the autogpt script with the command line arguments
$PY -m autogpt $@

read -p "Press any key to continue..."