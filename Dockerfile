# Use an official Python base image from the Docker Hub
FROM python:3.11-slim

# Install git
RUN apt-get -y update
RUN apt-get -y install git

# Set environment variables
ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create a non-root user and set permissions
RUN useradd --create-home user
WORKDIR /home/user
RUN chown user:user /home/appuser
USER user

# Copy the requirements.txt file and install the requirements
COPY --chown=user:user requirements-docker.txt .
RUN pip install --no-cache-dir --user -r requirements-docker.txt

# Copy the application files
COPY --chown=user:user autogpt/ ./autogpt

# Set the entrypoint
ENTRYPOINT ["python", "-m", "autogpt"]
