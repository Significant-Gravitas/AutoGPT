#!/bin/bash

cd llamafile-integration

# Download the mistral-7b-instruct llamafile from HuggingFace
wget -nc https://huggingface.co/jartine/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.llamafile
chmod +x mistral-7b-instruct-v0.2.Q5_K_M.llamafile
./mistral-7b-instruct-v0.2.Q5_K_M.llamafile --version
