# Use an official Python base image from the Docker Hub
FROM python:3.10-slim

# 'dev' or 'release' container build
ARG BUILD_TYPE=dev

# Install browsers
RUN apt-get update && apt-get install -y \
    chromium-driver firefox-esr \
    ca-certificates

# Install utilities
RUN apt-get install -y curl jq wget git

# Set environment variables
ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install the required python packages globally
ENV PATH="$PATH:/root/.local/bin"
COPY requirements.txt .

# Only install dev dependencies in dev container builds
RUN [ '${BUILD_TYPE}' = 'dev' ] || sed -i '/Items below this point will not be included in the Docker Image/,$d' requirements.txt && \
	pip install --no-cache-dir -r requirements.txt

# Copy the application files
WORKDIR /app
COPY autogpt/ ./autogpt

# Set the entrypoint
ENTRYPOINT ["python", "-m", "autogpt"]
