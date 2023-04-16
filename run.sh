#!/bin/bash
# This script checks if all the required packages are installed and installs them if not.

set -e # Exit immediately if a command exits with a non-zero status.
python scripts/check_requirements.py requirements.txt # Check if all the required packages are installed
if [ $? -eq 1 ]; then # If not, install them
echo "Installing missing packages..."
pip install -r requirements.txt
fi
python -m autogpt "$@" # Run the main script
read -n1 -rsp $'Press any key to continue...\n' # Wait for user input before exiting