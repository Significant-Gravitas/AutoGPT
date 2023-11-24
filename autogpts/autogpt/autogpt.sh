#!/usr/bin/env bash

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
    $PYTHON_CMD scripts/check_requirements.py
    if [ $? -eq 1 ]
    then
        echo
        $PYTHON_CMD -m poetry install --without dev
        echo
        echo "Finished installing packages! Starting AutoGPT..."
        echo
    fi
    $PYTHON_CMD -m autogpt "$@"
else
    echo "Python 3.10 or higher is required to run Auto GPT."
fi
