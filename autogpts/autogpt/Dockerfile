# 'dev' or 'release' container build
ARG BUILD_TYPE=dev

# Use an official Python base image from the Docker Hub
FROM python:3.10-slim AS autogpt-base

# Install browsers
RUN apt-get update && apt-get install -y \
    chromium-driver firefox-esr ca-certificates gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install utilities
RUN apt-get update && apt-get install -y \
    curl jq wget git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_NO_INTERACTION=1

# Install and configure Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN poetry config installer.max-workers 10

# Set the entrypoint
CMD ["poetry", "run", "--install-plugin-deps"]

# dev build -> include everything
FROM autogpt-base as autogpt-dev
RUN poetry install
WORKDIR /app
ONBUILD COPY . ./

# release build -> include bare minimum
FROM autogpt-base as autogpt-release
RUN poetry install --without dev,benchmark
WORKDIR /app
ONBUILD COPY autogpt/ ./autogpt
ONBUILD COPY scripts/ ./scripts
ONBUILD COPY plugins/ ./plugins
ONBUILD COPY prompt_settings.yaml ./prompt_settings.yaml
ONBUILD RUN mkdir ./data

FROM autogpt-${BUILD_TYPE} AS auto-gpt
