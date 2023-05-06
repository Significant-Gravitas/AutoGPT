#!/bin/bash

function usage() {
    echo "Usage: $0 [-h] [-v] [args]"
    echo "  -h  Display this help message"
    echo "  -v  Use Python virtual environment"
    exit 1
}

use_venv=false

while getopts "hv" opt; do
    case ${opt} in
        h )
            usage
            ;;
        v )
            use_venv=true
            ;;
        \? )
            echo "Invalid option: -$OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Option -$OPTARG requires an argument." 1>&2
            usage
            ;;
    esac
done
shift $((OPTIND -1))

function find_python_command() {
    if $use_venv && [ -n "$VIRTUAL_ENV" ]
    then
        echo "$VIRTUAL_ENV/bin/python"
    elif command -v python &> /dev/null
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

if $use_venv && [ -n "$VIRTUAL_ENV" ]
then
    echo "Using Python virtual environment: $VIRTUAL_ENV"
    source $VIRTUAL_ENV/bin/activate
fi

$PYTHON_CMD scripts/check_requirements.py requirements.txt
if [ $? -eq 1 ]
then
    echo Installing missing packages...
    $PYTHON_CMD -m pip install -r requirements.txt
fi
$PYTHON_CMD -m autogpt $@

if $use_venv && [ -n "$VIRTUAL_ENV" ]
then
    echo "Deactivating Python virtual environment: $VIRTUAL_ENV"
    deactivate
fi

read -p "Press any key to continue..."