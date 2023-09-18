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
        echo "Installing python3 using pyenv..."
        if ! command -v pyenv &> /dev/null
        then
            echo "pyenv could not be found"
            echo "Installing pyenv..."
            curl https://pyenv.run | bash
        fi
        pyenv install 3.11.5
        pyenv global 3.11.5
    fi

    if ! command -v poetry &> /dev/null
    then
        echo "poetry could not be found"
        echo "Installing poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
    fi
fi

if ! command -v flutter &> /dev/null
then
    echo "flutter could not be found"
    echo "Please install flutter following the instrutions found at https://docs.flutter.dev/get-started/install"
    exit 1
fi
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if ! command -v google-chrome-stable &> /dev/null
    then
        echo "Google Chrome could not be found"
        echo "Please install Google Chrome manually from https://www.google.com/chrome/"
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome &> /dev/null
    then
        echo "Google Chrome could not be found"
        echo "Please install Google Chrome manually from https://www.google.com/chrome/"
        exit 1
    fi
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    if ! command -v /c/Program\ Files\ \(x86\)/Google/Chrome/Application/chrome.exe &> /dev/null
    then
        echo "Google Chrome could not be found"
        echo "Please install Google Chrome manually from https://www.google.com/chrome/"
        exit 1
    fi
else
    echo "Unsupported OS. Please install Google Chrome manually from https://www.google.com/chrome/"
    exit 1
fi
