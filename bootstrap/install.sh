#!/usr/bin/env bash

# ====================================================================================================
# This script is used to install Auto-GPT via the Docker method
# If other installation methods are desired, refer to the documentation at https://docs.agpt.co/
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

# ====================================================================================================
# Configurable variables
# ====================================================================================================
CONFIG_DIR=~/.autogpt
DOCKER_IMAGE="significantgravitas/auto-gpt"

# Files to download from GitHub (use paths relative to the repository root)
# Any template files will be copied after download (.template will be removed)
FILES_TO_DOWNLOAD=(
    "docker-compose.yml.template"
    "plugins_config.yaml"
    "prompt_settings.yaml"
    ".env.template"
    "azure.yaml.template"
    "requirements.txt" # TODO: Will this allow users to install additional Python packages?
)

# This script will be moved to the user's PATH (/usr/local/bin)
# It will be renamed to 'autogpt' and act as the main entrypoint and user command
LAUNCHER_CMD_SRC="scripts/autogpt_cmd.sh"

# ====================================================================================================
# End of configurable variables
# ====================================================================================================

# Intro
echo "Hi, this script will install the 'autogpt' command on your system."
echo ""
echo "Note:"
echo "- If you intend to use Auto-GPT as a Python module, don't use this script."
echo "- Please refer to the documentation at https://docs.agpt.co/."
echo ""
echo "Press Ctrl+C to exit the script at any time."
echo ""

# Github user: --user or -u
# Github repo: --repo or -r
# Branch or tag: --branch, -b, --tag, -t
# Yes: --yes or -y
# Help: --help or -h
GITHUB_USER="Significant-Gravitas"
GITHUB_REPO="Auto-GPT"
BRANCH="stable"
HELP=0
YES=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -u|--user) GITHUB_USER="$2"; shift ;;
        -r|--repo) GITHUB_REPO="$2"; shift ;;
        -b|--branch|-t|--tag) BRANCH="$2"; shift ;;
        -y|--yes) YES=1; shift ;;
        -h|--help) HELP=1; shift ;;
        *) echo "Unknown parameter passed: $1"; echo ""; HELP=1 ;;
    esac
    shift
done

# Help
if [ $HELP == 1 ]; then
    echo "Usage: install.sh [OPTIONS]"
    echo "Options:"
    echo "  -u, --user <github_user>    GitHub user to download from (default: Significant-Gravitas)"
    echo "  -r, --repo <github_repo>    GitHub repo to download from (default: Auto-GPT)"
    echo "  -b, --branch <branch>       GitHub branch or tag to download from (default: stable)"
    echo "  -t, --tag <tag>             GitHub branch or tag to download from (default: stable)"
    echo "  -y, --yes                   Skip confirmation prompts"
    echo "  -h, --help                  Show this help message"
    exit 0
fi

pause() {
    if [ $YES == 0 ]; then 
        echo "Press any key to continue..."
        read -s -n 1
        echo ---
        echo
    fi
}
pause

# Construct the URL to the raw files:
GITHUB_FILES_BASE="https://raw.githubusercontent.com/$GITHUB_USER/$GITHUB_REPO/$BRANCH/"
echo "Using GitHub files base: $GITHUB_FILES_BASE"
echo ""

# Check if Docker is installed
echo "Checking if Docker is installed..."
echo ""

if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Docker is required to use this script."
    echo "For non-Docker options, please refer to the documentation at https://docs.agpt.co/."
    echo "If you have Docker Desktop installed, please make sure it is running and then try this script again."
    echo "You can set Docker Desktop to run on startup in the settings."
    echo ""

    if [ $YES == 0 ]; then
        read -p "Do you want me to try and install Docker now? (y/n): " -n 1 -r
        echo
        if [[ !$REPLY =~ ^[Yy]$|^$ ]]
        then
            echo "Docker is required to use this script. Exiting..."
            exit 1
        fi
    fi

    if [ $OS == "MAC" ]; then
        echo "Attempting to use Homebrew to install Docker on your Mac..."

        if ! command -v brew &> /dev/null
        then
            echo "Homebrew is not installed. Opening Docker website for manual installation... "
            echo "Note that you will need to make sure Docker Desktop running before you run this script again."
            open https://hub.docker.com/editions/community/docker-ce-desktop-mac
            exit 1
        fi

        brew install --cask docker
    else
        echo "Installing Docker on Linux..."
        echo "Some of the instructions in this section may require administrator privileges."
        echo "If prompted, please enter your password."
        echo "Enabling xtrace to show commands as they are run..."
        set -x
        sudo apt-get update
        sudo apt-get install apt-transport-https ca-certificates curl software-properties-common -y
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
        sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
        sudo apt-get update
        sudo apt-get install docker-ce -y
        echo "Done. Disabling xtrace..."
        set +x
    fi

    if ! command -v docker &> /dev/null
    then
        echo "Docker install failed. Please refer to the documentation at https://docs.agpt.co/ for non-Docker options."
        exit 1
    fi

    if [ $YES == 0 ]; then read -p "Docker is now installed. Press any key to continue installing Auto-GPT..."; echo; fi
else
    echo "Docker is installed!"
    echo
fi

pause

# Create the config directory and change to it
if [[ ! -d $CONFIG_DIR ]]; then
    echo "Creating config directory at $CONFIG_DIR..."
    echo "All user configuration files will be stored here."
    echo
    mkdir -p $CONFIG_DIR
fi

if [[ ! -d $CONFIG_DIR/bin]]; then
    echo "Creating bin directory at $CONFIG_DIR/bin..."
    echo "All user configuration files will be stored here."
    echo
    mkdir -p $CONFIG_DIR/bin
fi

cd $CONFIG_DIR

# Download files from GitHub
echo "Downloading Auto-GPT config files from GitHub..."
for file in "${FILES_TO_DOWNLOAD[@]}"; do
    echo "- Downloading $file. Source: $GITHUB_FILES_BASE$file ..."
    curl -O $GITHUB_FILES_BASE$file

    # 404?
    if [ $? -ne 0 ]; then
        echo "Error downloading $file. Please check the URL and try again."
        exit 1
    fi

    # 4XX in contents? Format: 4XX: [message]
    if grep -q "4[0-9][0-9]: [*]" $file; then
        echo "Error downloading $file. Please check the URL and try again."
        cat $file
        exit 1
    fi
    
    # Create config files from templates.
    if [[ $file == *.template ]]; then
        echo "Copying $file to ${file%.*}..."
        cp $file ${file%.*}
    fi
done

echo "Files downloaded."; echo; pause

# Open AI Key
echo "Please enter your Open AI API key. You can find it at https://platform.openai.com/account/api-keys."
read -s -p "Open API Key: " OPENAI_API_KEY
echo
sed -i '' "s/your-openai-api-key/$OPENAI_API_KEY/g" .env
echo "Open AI API key saved to .env file."
echo

# Pull Docker image
echo "Pulling Docker image ($DOCKER_IMAGE)..."
echo
docker pull $DOCKER_IMAGE

# Install launch script
LAUNCHER_CMD_SRC=$GITHUB_FILES_BASE$LAUNCHER_CMD_SRC
echo "Creating launch script. Source: $LAUNCHER_CMD_SRC)..."
echo
curl -o ./bin/autogpt $LAUNCHER_CMD_SRC

# 404?
if [ $? -ne 0 ]; then
    echo "Error downloading $LAUNCHER_CMD_SRC. Please check the URL and try again."
    exit 1
fi

# 4XX in contents? Format: 4XX: [message]
if grep -q "4[0-9][0-9]: [*]" ./bin/autogpt; then
    echo "Error downloading $LAUNCHER_CMD_SRC. Please check the URL and try again."
    cat ./bin/autogpt
    exit 1
fi

chmod +x ./bin/autogpt

# Install complete
echo
echo "Installation complete!"
echo 
echo "You can launch Auto-GPT by running 'autogpt' in your terminal."
echo
 
# Launch Auto-GPT
if [ $YES == 0 ]; then echo "Launch Auto-GPT now? (y/n)"
    read -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$|^$ ]]
    then
        exit 0
    fi
fi

echo
echo "Launching Auto-GPT..."
autogpt
exit 0