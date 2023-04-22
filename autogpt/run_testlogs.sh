#!/bin/bash

# Run the testlogs.py script
python testlogs.py

# Compile the .typ file to a PDF using typst
typst compile ../logs/log.typ
