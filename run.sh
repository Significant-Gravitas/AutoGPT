#!/bin/bash

# This script is used to run the main script in a virtual environment.

set_venv() {
    if [ -d "venv" ]; then # Check if the virtual environment exists
        source venv/bin/activate # Activate the virtual environment
    else
        echo "venv is not found. Creating venv..."
        create_venv
    fi
}

create_venv() {
    python3 -m venv venv && # Create a virtual environment
    source venv/bin/activate # Activate the virtual environment
}

{ # try
    set_venv # Set the virtual environment
} || { # catch
    echo "An error occurred while creating the virtual environment."
    echo "venv is not found. Installing venv..."
    sudo apt-get update
    sudo apt-get install python3-venv
    set_venv
}

set -e # Exit immediately if a command exits with a non-zero status.
# Check if all the required packages are installed

install_packages() {
    echo "Installing missing packages..."
    pip3 install -r requirements.txt
}

sudo apt install python3-pip
{ # try
    install_packages # Install the required packages
} || { # catch
    echo "An error occurred while installing the required packages."
    sudo apt-get update
    sudo apt-get install python3-pip
    install_packages # Install the required packages
}

python3 -m autogpt "$@" # Run the main script
read -n1 -rsp $'Press any key to continue...\n' # Wait for user input before exiting