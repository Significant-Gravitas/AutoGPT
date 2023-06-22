#!/usr/bin/env

# ====================================================================================================
# This script is used to launch Auto-GPT from the command line
# It is installed to /usr/local/bin/autogpt by the install.sh script
# ====================================================================================================

# This script shoudl not be run inside a Docker container
if [ -f /.dockerenv ]; then
    echo "This script should not be run inside a Docker container."
    exit 1
fi

# Allow github user to be changed using --user or -u argument
GITHUB_USER="Significant-Gravitas"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -u|--user) GITHUB_USER="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Allow repo to be changed using "--repo" or "-r" argument
GITHUB_REPO="Auto-GPT"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--repo) GITHUB_REPO="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Allow branch to be changed using "--branch, -b, --tag, -t" argument
# Default to "stable" branch
BRANCH="stable"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--branch|-t|--tag) BRANCH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Construct the URL to the raw files:
GITHUB_FILES_BASE="https://raw.githubusercontent.com/$GITHUB_USER/$GITHUB_REPO/$BRANCH/"

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
    curl -s $GITHUB_FILES_BASE/scripts/install.sh -u $GITHUB_USER -r $GITHUB_REPO -b $BRANCH | bash
    exit 0
fi

# Default behaviour: Launch Auto-GPT
docker compose run --rm auto-gpt $@