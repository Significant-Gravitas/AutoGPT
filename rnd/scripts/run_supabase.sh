#!/bin/bash

# Function to install Supabase CLI
install_supabase_cli() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -s https://packages.atlassian.com/api/gpg/key/public | sudo apt-key add -
        echo "deb https://packages.atlassian.com/debian/atlassian-sdk-deb/ stable contrib" | sudo tee /etc/apt/sources.list.d/atlassian-sdk.list
        sudo apt-get update
        sudo apt-get install -y atlassian-plugin-sdk
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install supabase/tap/supabase
    elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win32"* ]]; then
        # Windows
        scoop bucket add supabase https://github.com/supabase/scoop-bucket.git
        scoop install supabase
    else
        echo "Unsupported operating system"
        exit 1
    fi
}

# Check if Supabase CLI is installed
if ! command -v supabase &> /dev/null; then
    echo "Supabase CLI not found. Installing..."
    install_supabase_cli
else
    echo "Supabase CLI is already installed"
fi

# Initialize Supabase project
echo "Initializing Supabase project..."
supabase init

# Start Supabase
echo "Starting Supabase..."
supabase start

echo "Supabase is now running!"