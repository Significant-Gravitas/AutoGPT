#!/bin/bash
# Local runner for Playbook Client

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Playbook Client..."
python3 app_client.py
