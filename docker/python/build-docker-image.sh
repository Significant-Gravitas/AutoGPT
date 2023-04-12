#!/usr/bin/env bash

# Script to quickly rebuild and run the app in a Docker container.
# This file is run manually and its image is used by execute_python_file command

docker build . -t fsamir/python:3.11 -f python.Dockerfile

#Should print the json response:
docker run --rm -it \
           --memory=1200M \
           --name agpt-python \
           -v /Users/franklin/projects/learn/Auto-GPT/auto_gpt_workspace/:/worspace \
           fsamir/python:3.11 \
           python /worspace/book_scraper.py
