#!/usr/bin/env bash

function find_python_command() {
    if command -v python3 &> /dev/null
    then
        echo "python3"
    elif command -v python &> /dev/null
    then
        echo "python"
    else
        echo "Python not found. Please install Python."
        exit 1
    fi
}

PYTHON_CMD=$(find_python_command)

if $PYTHON_CMD -c "import sys; sys.exit(sys.version_info < (3, 10))"; then
    if ! $PYTHON_CMD scripts/check_requirements.py; then
        echo
        poetry install --without dev
        echo
        echo "Finished installing packages! Starting AutoGPT..."
        echo
    fi
    poetry run autogpt "$@"
else
    echo "Python 3.10 or higher is required to run Auto GPT."
fi
