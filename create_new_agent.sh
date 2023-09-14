#!/bin/bash

if [ "$1" == "--help" ]; then
  echo "Usage: $0 new_agent_name"
  echo "This script needs an agent name (no spaces) and it will create a new agent in the autogpts folder."
  exit 0
fi

if [ -z "$1" ]; then
  echo "Error: No agent name provided. Run $0 --help for usage."
  exit 1
fi

new_name=$1
cp -r autogpts/forge autogpts/$new_name

echo "New agent $new_name created and switched to the new directory in autogpts folder."

