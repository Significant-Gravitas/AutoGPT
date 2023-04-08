#!/usr/bin/env bash

# Script to quickly rebuild and run the app in a Docker container.
# This file is run manually and its image is used by execute_python_file command

docker build . -t fsamir/python:3.11 -f python.Dockerfile