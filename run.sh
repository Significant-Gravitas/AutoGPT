#!/bin/bash

# Bash
COLOR_NC=$(tput sgr0)
COLOR_B=$(tput setaf 0); COLOR_R=$(tput setaf 1); COLOR_G=$(tput setaf 2); COLOR_Y=$(tput setaf 3); COLOR_BL=$(tput setaf 4); COLOR_P=$(tput setaf 5); COLOR_C=$(tput setaf 6); COLOR_W=$(tput setaf 7)
COLOR_BB=$(tput setaf 8); COLOR_BR=$(tput setaf 9); COLOR_BG=$(tput setaf 10); COLOR_BY=$(tput setaf 11); COLOR_BBL=$(tput setaf 12); COLOR_BP=$(tput setaf 13); COLOR_BC=$(tput setaf 14); COLOR_BW=$(tput setaf 15)
COLOR_UB=$(tput setaf 16); COLOR_UR=$(tput setaf 17); COLOR_UG=$(tput setaf 18); COLOR_UY=$(tput setaf 19); COLOR_UBL=$(tput setaf 20); COLOR_UP=$(tput setaf 21); COLOR_UC=$(tput setaf 22); COLOR_UW=$(tput setaf 23)
COLOR_OB=$(tput setab 0); COLOR_OR=$(tput setab 1); COLOR_OG=$(tput setab 2); COLOR_OY=$(tput setab 3); COLOR_OBL=$(tput setab 4); COLOR_OP=$(tput setab 5); COLOR_OC=$(tput setab 6); COLOR_OW=$(tput setab 7)
COLOR_IB=$(tput setaf 24); COLOR_IR=$(tput setaf 25); COLOR_IG=$(tput setaf 26); COLOR_IY=$(tput setaf 27); COLOR_IBL=$(tput setaf 28); COLOR_IP=$(tput setaf 29); COLOR_IC=$(tput setaf 30); COLOR_IW=$(tput setaf 31)
COLOR_BIB=$(tput setaf 32); COLOR_BIR=$(tput setaf 33); COLOR_BIG=$(tput setaf 34); COLOR_BIY=$(tput setaf 35); COLOR_BIBL=$(tput setaf 36); COLOR_BIP=$(tput setaf 37); COLOR_BIC=$(tput setaf 38); COLOR_BIW=$(tput setaf 39)
COLOR_OIB=$(tput setab 8); COLOR_OIR=$(tput setab 9); COLOR_OIG=$(tput setab 10); COLOR_OIY=$(tput setab 11); COLOR_OIBL=$(tput setab 12); COLOR_OIP=$(tput setab 13); COLOR_OIC=$(tput setab 14); COLOR_OIW=$(tput setab 15)

# Usage instructions
function usage() {
    printf "${COLOR_G}Usage:${COLOR_NC} %s [-h] [-v] [-e VENV_PATH] [-c] [-s] [-C AI_SETTINGS] [-l CONTINUOUS_LIMIT] [-S] [-d] [-g] [-G] [-m MEMORY_TYPE] [-b BROWSER_NAME] [-A] [-N] [-w WORKSPACE_DIR] [-I] [args]\n" "$0"
    printf "\n${COLOR_Y}Command-line Options:${COLOR_NC}\n\n"
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-h, --help" "Show this help message and exit."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-v, --virtualenv" "Use Python virtual environment. (default: false)"
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-e, --venv-path VENV_PATH" "Set the path to the Python virtual environment."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-c, --continuous" "Enable continuous mode for autogpt. (default: false)"
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-s, --skip-reprompt" "Skip the re-prompting messages at the beginning of the script."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-C, --ai-settings AI_SETTINGS" "Specify which ai_settings.yaml file to use."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-l, --continuous-limit CONTINUOUS_LIMIT" "Define the number of times to run in continuous mode."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-S, --speak" "Enable Speak Mode."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-d, --debug" "Enable Debug Mode."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-g, --gpt3only" "Enable GPT3.5 Only Mode."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-G, --gpt4only" "Enable GPT4 Only Mode."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-m, --use-memory MEMORY_TYPE" "Define which Memory backend to use."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-b, --browser-name BROWSER_NAME" "Specify which web-browser to use when using selenium to scrape the web."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-A, --allow-downloads" "Dangerous: Allows Auto-GPT to download files natively."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-N, --skip-news" "Specifies whether to suppress the output of latest news on startup."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-w, --workspace-directory WORKSPACE_DIR" "Specifies which directory to use for the workspace."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "-I, --install-plugin-deps" "Installs external dependencies for 3rd party plugins."
    printf "\n${COLOR_Y}Environment Variables:${COLOR_NC}\n\n  ${COLOR_BL}You can set these values instead of passing command line arguments,\n  before executing this script.${COLOR_NC}\n\n"
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "USE_VENV" "Set to true to use Python virtual environment. (default: false)"
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "VENV_PATH" "Set the path to the Python virtual environment."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "CONTINUOUS" "Set to true to enable continuous mode for autogpt. (default: false)"
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "SKIP_REPROMPT" "Set to true to skip the re-prompting messages at the beginning of the script."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "AI_SETTINGS" "Specify which ai_settings.yaml file to use."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "CONTINUOUS_LIMIT" "Define the number of times to run in continuous mode."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "SPEAK" "Set to true to enable Speak Mode."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "DEBUG" "Set to true to enable Debug Mode."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "GPT3_ONLY" "Set to true to use GPT-3 API only for autogpt. (default: false)"
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "GPT4_ONLY" "Set to true to use GPT-4 API only for autogpt. (default: false)"
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "MEMORY_TYPE" "Define which Memory backend to use."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "BROWSER_NAME" "Specify which web-browser to use when using selenium to scrape the web."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "ALLOW_DOWNLOADS" "Set to true to allow Auto-GPT to download files natively. (default: false)"
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "SKIP_NEWS" "Set to true to suppress the output of latest news on startup. (default: false)"
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "WORKSPACE_DIRECTORY" "Specifies which directory to use for the workspace."
    printf "  ${COLOR_C}%-40s${COLOR_NC}%s\n" "INSTALL_PLUGIN_DEPS" "Set to true to install external dependencies for 3rd party plugins. (default: false)"
    exit 1
}

# Parse Command-line arguments
function parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            -h|--help) usage;;
            -v|--virtualenv) USE_VENV=true;;
            -e|--venv-path) export VENV_PATH="$2"; shift ;;
            -c|--continuous) CONTINUOUS=true;;
            -s|--skip-reprompt) SKIP_REPROMPT=true;;
            -C|--ai-settings) AI_SETTINGS="$2"; shift ;;
            -l|--continuous-limit) CONTINUOUS_LIMIT="$2"; shift ;;
            -S|--speak) SPEAK=true;;
            -d|--debug) DEBUG=true;;
            -g|--gpt3only) GPT3_ONLY=true;;
            -G|--gpt4only) GPT4_ONLY=true;;
            -m|--use-memory) MEMORY_TYPE="$2"; shift ;;
            -b|--browser-name) BROWSER_NAME="$2"; shift ;;
            -A|--allow-downloads) ALLOW_DOWNLOADS=true;;
            -N|--skip-news) SKIP_NEWS=true;;
            -w|--workspace-directory) WORKSPACE_DIR="$2"; shift ;;
            -I|--install-plugin-deps) INSTALL_PLUGIN_DEPS=true;;
            *) break;;
        esac
        shift
    done
}

function build_command() {
    COMMAND="$PYTHON_CMD -m autogpt"
    if $CONTINUOUS; then
        COMMAND+=" --continuous"
    fi
    if $GPT3_ONLY; then
        COMMAND+=" --gpt3only"
    fi
    if $GPT4_ONLY; then
        COMMAND+=" --gpt4only"
    fi
    if $SKIP_REPROMPT; then
        COMMAND+=" --skip-reprompt"
    fi
    if [ -n "$AI_SETTINGS" ]; then
        COMMAND+=" --ai-settings $AI_SETTINGS"
    fi
    if [ -n "$CONTINUOUS_LIMIT" ]; then
        COMMAND+=" --continuous-limit $CONTINUOUS_LIMIT"
    fi
    if $SPEAK; then
        COMMAND+=" --speak"
    fi
    if $DEBUG; then
        COMMAND+=" --debug"
    fi
    if [ -n "$MEMORY_TYPE" ]; then
        COMMAND+=" --use-memory $MEMORY_TYPE"
    fi
    if [ -n "$BROWSER_NAME" ]; then
        COMMAND+=" --browser-name $BROWSER_NAME"
    fi
    if $ALLOW_DOWNLOADS; then
        COMMAND+=" --allow-downloads"
    fi
    if $SKIP_NEWS; then
        COMMAND+=" --skip-news"
    fi
    if [ -n "$WORKSPACE_DIR" ]; then
        COMMAND+=" --workspace-directory $WORKSPACE_DIR"
    fi
    if $INSTALL_PLUGIN_DEPS; then
        COMMAND+=" --install-plugin-deps"
    fi
    COMMAND+=" $@"
    echo "$COMMAND"
}

# Set default values for variables if not defined in the environment
USE_VENV=${USE_VENV:-false}
VENV_PATH=${VENV_PATH:-./venv}
CONTINUOUS=${CONTINUOUS:-false}
SKIP_REPROMPT=${SKIP_REPROMPT:-false}
AI_SETTINGS=${AI_SETTINGS:-./ai_settings.yaml}
CONTINUOUS_LIMIT=${CONTINUOUS_LIMIT:-50}
SPEAK=${SPEAK:-false}
DEBUG=${DEBUG:-false}
GPT3_ONLY=${GPT3_ONLY:-false}
GPT4_ONLY=${GPT4_ONLY:-false}
MEMORY_TYPE=${MEMORY_TYPE:-redis}
BROWSER_NAME=${BROWSER_NAME:-chrome}
ALLOW_DOWNLOADS=${ALLOW_DOWNLOADS:-false}
SKIP_NEWS=${SKIP_NEWS:-false}
WORKSPACE_DIR=${WORKSPACE_DIR:-./autogpt/auto_gpt_workspace}
INSTALL_PLUGIN_DEPS=${INSTALL_PLUGIN_DEPS:-false}

parse_args "$@"

# Use virtual environment if requested
if $USE_VENV && [ -n "$VENV_PATH" ]; then
echo "Using Python virtual environment: $VENV_PATH"
source "$VENV_PATH"/bin/activate
fi

# Check and install missing requirements
PYTHON_CMD=$(command -v python3 || command -v python || { echo "Python not found. Please install Python." >&2; exit 1; })
$PYTHON_CMD scripts/check_requirements.py requirements.txt
if [ $? -eq 1 ]; then
echo Installing missing packages...
$PYTHON_CMD -m pip install -r requirements.txt
fi

# Run the command
COMMAND=$(build_command "$@")
eval "$COMMAND"

# Deactivate virtual environment if used
if $USE_VENV && [ -n "$VENV_PATH" ]; then
    echo "Deactivating Python virtual environment: $VENV_PATH"
    deactivate
fi

read -p "Press any key to continue..."
