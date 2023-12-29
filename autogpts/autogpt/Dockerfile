# 'dev' or 'release' container build
ARG BUILD_TYPE=dev

# Use an official Python base image from the Docker Hub
FROM python:3.10-slim AS autogpt-base

# Install browsers
RUN apt-get update && apt-get install -y \
    chromium-driver ca-certificates gcc \
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
    POETRY_VIRTUALENVS_PATH="/venv" \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_NO_INTERACTION=1

# Install and configure Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN poetry config installer.max-workers 10

WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Set the entrypoint
ENTRYPOINT ["poetry", "run", "autogpt"]
CMD []

# dev build -> include everything
FROM autogpt-base as autogpt-dev
RUN poetry install --no-cache --no-root \
    && rm -rf $(poetry env info --path)/src
ONBUILD COPY . ./

# release build -> include bare minimum
FROM autogpt-base as autogpt-release
RUN poetry install --no-cache --no-root --without dev \
    && rm -rf $(poetry env info --path)/src
ONBUILD COPY autogpt/ ./autogpt
ONBUILD COPY scripts/ ./scripts
ONBUILD COPY plugins/ ./plugins
ONBUILD COPY prompt_settings.yaml ./prompt_settings.yaml
ONBUILD COPY README.md ./README.md
ONBUILD RUN mkdir ./data

FROM autogpt-${BUILD_TYPE} AS autogpt
RUN poetry install --only-root
