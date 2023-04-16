#!/bin/bash

# This script is used to run the main script in a virtual environment.
if ! command -v python3 -m venv &> /dev/null    # Check if venv is installed and install it if not
then
echo "venv is not found. Installing venv..."
sudo apt-get update
sudo apt-get install python3-venv
fi

python3 -m venv env # Create a virtual environment
source env/bin/activate # Activate the virtual environment

set -e # Exit immediately if a command exits with a non-zero status.
python scripts/check_requirements.py requirements.txt # Check if all the required packages are installed
if [ $? -eq 1 ]; then # If not, install them
echo "Installing missing packages..."
pip install -r requirements.txt
fi
python -m autogpt "$@" # Run the main script
read -n1 -rsp $'Press any key to continue...\n' # Wait for user input before exiting