#!/bin/bash
python scripts/check_requirements.py requirements.txt
python -m autogpt $@
read -p "Press any key to continue..."
