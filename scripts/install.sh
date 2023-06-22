#!/usr/bin/env bash

# ====================================================================================================
# This script is used to install Auto-GPT via the Docker method
# If other installation methods are desired, refer to the documentation at https://docs.agpt.co/
# ====================================================================================================

# Definitions
CONFIG_DIR=~/.autogpt
DOCKER_IMAGE_NAME="significantgravitas/auto-gpt"

# Files to download from GitHub (use paths relative to the repository root)
# Any template files will be copied after download (.template will be removed)
GITHUB_FILE_DOWNLOAD_LIST=(
    "docker-compose.yaml"
    "plugins_config.yaml"
    "prompt_settings.yaml"
    ".env.template"
    "azure.yaml.template"
    "requirements.txt" # This allows users to modify the requirements.txt file to install additional Python packages
)

# This script will be moved to the user's PATH (/usr/local/bin)
# It will be renamed to 'autogpt' and act as the main entrypoint and user command
LAUNCHER_SCRIPT="scripts/autogpt_cmd.sh"

# This script should not be run inside a Docker container
if [ -f /.dockerenv ]; then
    echo "This script should not be run inside a Docker container."
    exit 1
fi

# Intro
echo "Hi, this is Auto-GPT. Welcome to my installation script!"
echo "This script will install the 'autogpt' command on your system. It is designed to work on Linux and MacOS."
echo "Note: If you intend to use Auto-GPT as a Python module in your project, this script is not for you. Please refer to the documentation at https://docs.agpt.co/."
echo "Press Ctrl+C to exit the script at any time."
echo ""

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

# Check the OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="LINUX"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="MAC"
else
    echo "This script is only compatible with Linux and MacOS."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Docker is required to use this script. For non-Docker options, please refer to the documentation at https://docs.agpt.co/."
    echo "If you have Docker Desktop installed, please make sure it is running and then try this script again. You can set Docker Desktop to run on startup in the settings."

    read -p "Do you want me to try and install Docker now? (y/n): " -n 1 -r
    echo
    if [[ !$REPLY =~ ^[Yy]$|^$ ]]
    then
        echo "Docker is required to use this script. Exiting..."
        exit 1
    fi

    if [ $OS == "MAC" ]; then
        echo "Attempting to use Homebrew to install Docker on your Mac..."

        if ! command -v brew &> /dev/null
        then
            echo "Homebrew is not installed. Opening Docker website for manual installation... Note that you will need to make sure Docker Desktop running before you run this script again."
            open https://hub.docker.com/editions/community/docker-ce-desktop-mac
            exit 1
        fi

        brew install --cask docker
    else
        echo "Installing Docker on Linux..."
        sudo apt-get update
        sudo apt-get install apt-transport-https ca-certificates curl software-properties-common -y
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
        sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
        sudo apt-get update
        sudo apt-get install docker-ce -y
    fi
else
    echo "Docker is already installed."
fi

# Create the config directory and change to it
if [[ ! -d $CONFIG_DIR ]]; then
    echo "Creating config directory at $CONFIG_DIR..."
    echo "All user configuration files will be stored here."
    mkdir -p $CONFIG_DIR
fi
cd $CONFIG_DIR

# Download files from GitHub
echo "Downloading Auto-GPT config files from GitHub..."
for file in "${GITHUB_DOWNLOAD_LIST[@]}"; do
    curl -O $GITHUB_FILES_BASE/$file

    # Create config files from templates
    if [[ $file == *.template && ! -f ${file%.*} ]]; then
        echo "Copying $file to ${file%.*}..."
        cp $file ${file%.*}
    fi
done

# Open AI Key
echo "Please enter your Open AI API key. You can find it at https://platform.openai.com/account/api-keys."
read -s -p "Open API Key: " OPENAI_API_KEY
echo
sed -i '' "s/your-openai-api-key/$OPENAI_API_KEY/g" .env
echo "Open AI API key saved to .env file."

# Pull Docker image
echo "Pulling Docker image..."
docker pull $DOCKER_IMAGE_NAME

# Install launch script
echo "Creating launch script..."
curl -o autogpt $GITHUB_FILES_BASE/$LAUNCH_SCRIPT
chmod +x autogpt
mv autogpt /usr/local/bin/

# Launch Auto-GPT
echo "Launching the application..."
/usr/local/bin/autogpt

echo "Installation complete! You can start using Auto-GPT by running 'autogpt' in your terminal."