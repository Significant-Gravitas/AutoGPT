#!/usr/bin/env bash

# ====================================================================================================
# This script is used to launch Auto-GPT from the command line
# It is installed to /usr/local/bin/autogpt by the install.sh script
# ====================================================================================================

# This script should not be run inside a Docker container
if [ -f /.dockerenv ]; then
    echo "This script should not be run inside a Docker container."
    exit 1
fi

# Check the OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="LINUX"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="MAC"
else
    echo "This script is only compatible with Linux and MacOS."
    exit 1
fi

# Github user: --user or -u
# Github repo: --repo or -r
# Branch or tag: --branch, -b, --tag, -t
# Rebuild: --rebuild
# Upgrade: --upgrade
# Reinstall: --reinstall
GITHUB_USER="Significant-Gravitas"
GITHUB_REPO="Auto-GPT"
BRANCH="stable"
REBUILD=0
UPGRADE=0
REINSTALL=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -u|--user) GITHUB_USER="$2"; shift ;;
        -r|--repo) GITHUB_REPO="$2"; shift ;;
        -b|--branch|-t|--tag) BRANCH="$2"; shift ;;
        --rebuild) REBUILD=1; shift ;;
        --upgrade) UPGRADE=1; shift ;;
        --reinstall) REINSTALL=1; shift ;;
        *) echo "";;
    esac
    shift
done

# Construct the URL to the raw files:
GITHUB_FILES_BASE="https://raw.githubusercontent.com/$GITHUB_USER/$GITHUB_REPO/$BRANCH/"

CONFIG_DIR=~/.autogpt
cd $CONFIG_DIR

# If the user has changed requirements.txt, they can run autogpt --rebuild to rebuild the Docker image
if [ REBUILD == 1 ]; then
    echo "Rebuilding Auto-GPT image..."
    docker compose build auto-gpt
    exit 0
fi

# Upgrade hook --upgrade
if [ UPGRADE == 1 ]; then
    echo "Upgrading Auto-GPT image..."
    docker compose pull auto-gpt
    exit 0
fi

# Reinstall hook --reinstall
if [ REINSTALL == 1 ]; then
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
    BACKUP_DIR=$CONFIG_DIR-backup-$(date +%Y-%m-%d_%H-%M-%S)
    mkdir -p $BACKUP_DIR
    cp -r $CONFIG_DIR/* $BACKUP_DIR

    # Delete config files
    echo "Deleting config files..."
    ## !!!WARNING!!! TODO: Decide whether to use -rf here !!!WARNING!!!
    # rm -rf $CONFIG_DIR
    rm -r $CONFIG_DIR

    # Reinstall
    echo "Reinstalling Auto-GPT..."
    curl -sSL $GITHUB_FILES_BASE/scripts/install.sh | bash -s -- -u $GITHUB_USER -r $GITHUB_REPO -b $BRANCH
    exit 0
fi

# Default behaviour: Launch Auto-GPT
docker compose run --rm auto-gpt $@