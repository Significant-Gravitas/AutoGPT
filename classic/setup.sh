#!/bin/bash

if [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "This script cannot be run on Windows."
    echo "Please follow the installation instructions at https://docs.python.org/3/using/windows.html"
    echo "To install poetry on Windows, please follow the instructions at https://python-poetry.org/docs/master/#installation"
    
    exit 1
else
    if ! command -v python3 &> /dev/null
    then
        echo "python3 could not be found"
        echo "Install python3 using pyenv ([y]/n)?"
        read response
        if [[ "$response" == "y" || -z "$response" ]]; then
            echo "Installing python3..."
            if ! command -v pyenv &> /dev/null
            then
                echo "pyenv could not be found"
                echo "Installing pyenv..."
                curl https://pyenv.run | bash
            fi
            pyenv install 3.11.5
            pyenv global 3.11.5
        else
            echo "Aborting setup"
            exit 1
        fi
    fi

    if ! command -v poetry &> /dev/null
    then
        echo "poetry could not be found"
        echo "Install poetry using official installer ([y]/n)?"
        read response
        if [[ "$response" == "y" || -z "$response" ]]; then
            echo "Installing poetry..."
            curl -sSL https://install.python-poetry.org | python3 -
        else
            echo "Aborting setup"
            exit 1
        fi
    fi
fi
