# Use an official Python base image from the Docker Hub
FROM python:3.10

# Declare working directory
WORKDIR /workspace/Auto-GPT

# Install browsers
RUN apt-get update && apt-get install -y \
    chromium-driver firefox-esr \
    ca-certificates

# Install utilities
RUN apt-get install -y curl jq wget git

# Install the required python packages globally
ENV PATH="$PATH:/root/.local/bin"
COPY ./requirements.txt .
