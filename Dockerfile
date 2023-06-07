# 'dev' or 'release' container build
ARG BUILD_TYPE=dev

# Use an official Python base image from the Docker Hub
FROM python:3.10-slim AS autogpt-base

# Install dependencies
RUN apt-get -y update \
    && apt-get -y install git curl jq wget chromium-driver ca-certificates firefox-esr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install the required python packages globally
ENV PATH="$PATH:/root/.local/bin"
COPY requirements.txt .

# Set the entrypoint
ENTRYPOINT ["python", "-m", "autogpt", "--install-plugin-deps"]

# dev build -> include everything
FROM autogpt-base as autogpt-dev
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
ONBUILD COPY . ./

# release build -> include bare minimum
FROM autogpt-base as autogpt-release
RUN sed -i '/Items below this point will not be included in the Docker Image/,$d' requirements.txt && \
	pip install --no-cache-dir -r requirements.txt
WORKDIR /app
ONBUILD COPY autogpt/ ./autogpt
ONBUILD COPY scripts/ ./scripts
ONBUILD COPY plugins/ ./plugins
ONBUILD RUN mkdir ./data

FROM autogpt-${BUILD_TYPE} AS auto-gpt
