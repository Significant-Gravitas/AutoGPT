# Use an official Python base image from the Docker Hub
FROM python:3.11-slim

# Install packages
RUN apt-get update && apt-get install -y curl jq git

# Set environment variables
ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install the required python packages globally
ENV PATH="$PATH:/root/.local/bin"
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Create a non-root user
RUN useradd -u 1000 appuser
USER appuser

# Copy the application files
WORKDIR /app
COPY --chown=appuser:appuser autogpt/ ./autogpt

# Set the entrypoint
ENTRYPOINT ["python", "-m", "autogpt"]
