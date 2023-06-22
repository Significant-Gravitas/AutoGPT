#!/usr/bin/env

# This script is used to launch Auto-GPT from the command line
# It is installed to /usr/local/bin/autogpt by the install.sh script

# If the user has changed requirements.txt, they can run autogpt --rebuild to rebuild the Docker image
if [ "$1" == "--rebuild" ]; then
    echo "Rebuilding Auto-GPT image..."
    docker compose build auto-gpt
    exit 0
fi

# Upgrade hook --upgrade
if [ "$1" == "--upgrade" ]; then
    echo "Upgrading Auto-GPT image..."
    docker compose pull auto-gpt
    exit 0
fi

# Reinstall hook --reinstall
if [ "$1" == "--reinstall" ]; then
    # Warning, this will delete the config files as well
    # Get user confirmation first!
    echo "This will delete all Auto-GPT config files and reinstall Auto-GPT."
    read -p "Are you sure you want to continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$|^$ ]]
    then
        echo "Exiting..."
        exit 1
    fi

    ## Backup config files
    echo "Backing up config files..."
    mkdir -p ~/.autogpt-backup-$(date +%Y-%m-%d_%H-%M-%S)
    cp -r ~/.autogpt/* ~/.autogpt-backup-$(date +%Y-%m-%d_%H-%M-%S)

    # Delete config files
    echo "Deleting config files..."
    ## !!!WARNING!!! TODO: Decide whether to use -rf here !!!WARNING!!!
    # rm -rf ~/.autogpt
    rm -r ~/.autogpt

    # Reinstall
    echo "Reinstalling Auto-GPT..."
    curl -s https://raw.githubusercontent.com/Significant-Gravitas/Auto-GPT/master/scripts/install.sh | bash
    exit 0
fi

# Default behaviour: Launch Auto-GPT
docker compose run --rm auto-gpt $@