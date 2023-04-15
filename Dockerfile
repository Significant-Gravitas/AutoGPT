# Use an official Python base image from the Docker Hub
FROM python:3.11-slim

# Install git
RUN apt-get -y update
RUN apt-get -y install git

# Set environment variables
ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Create a non-root user and set permissions
RUN useradd --create-home appuser
WORKDIR /home/appuser
RUN chown appuser:appuser /home/appuser


# Install Curl
RUN apt-get update && apt-get -y install curl

# Install and configure Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN poetry config installer.max-workers 10

# Install dependencies
COPY --chown=appuser:appuser pyproject.toml poetry.lock ./
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy the application files
COPY --chown=appuser:appuser autogpt/ ./autogpt/

# Switch to the non-root user
USER appuser

# Set the entrypoint and command
ENTRYPOINT ["poetry", "run"]
CMD ["python", "-m", "autogpt"]
