#!/usr/bin/env bash

# Go to autogpt/scripts/llamafile/
cd "$(dirname "$0")"

# Download the mistral-7b-instruct llamafile from HuggingFace
echo "Downloading mistral-7b-instruct-v0.2..."
wget -nc https://huggingface.co/jartine/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.llamafile
chmod +x mistral-7b-instruct-v0.2.Q5_K_M.llamafile
./mistral-7b-instruct-v0.2.Q5_K_M.llamafile --version

echo
echo "NOTE: To use other models besides mistral-7b-instruct-v0.2," \
     "download them into autogpt/scripts/llamafile/"
