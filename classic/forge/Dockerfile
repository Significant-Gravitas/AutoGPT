# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster as base

# Set work directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y build-essential curl ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install Poetry - respects $POETRY_VERSION & $POETRY_HOME
ENV POETRY_VERSION=1.1.8 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="$POETRY_HOME/bin:$PATH"

RUN pip3 install poetry

COPY pyproject.toml poetry.lock* /app/

# Project initialization:
RUN poetry install --no-interaction --no-ansi

ENV PYTHONPATH="/app:$PYTHONPATH"

FROM base as dependencies

# Copy project
COPY . /app


# Make port 80 available to the world outside this container
EXPOSE 8000

# Run the application when the container launches
CMD ["poetry", "run", "python", "classic/original_autogpt/__main__.py"]