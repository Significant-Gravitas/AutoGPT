# Use an official Python base image from the Docker Hub
FROM python:3.10-slim

# Install git
RUN apt-get -y update
RUN apt-get -y install git chromium-driver

# Install Xvfb and other dependencies for headless browser testing
RUN apt-get update \
    && apt-get install -y wget gnupg2 libgtk-3-0 libdbus-glib-1-2 dbus-x11 xvfb ca-certificates

# Install Firefox / Chromium
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y chromium firefox-esr

# Set environment variables
ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create a non-root user and set permissions
RUN useradd --create-home appuser
WORKDIR /home/appuser
RUN chown appuser:appuser /home/appuser
USER appuser

# Copy the requirements.txt file and install the requirements
COPY --chown=appuser:appuser requirements.txt .
RUN sed -i '/Items below this point will not be included in the Docker Image/,$d' requirements.txt && \
	pip install --no-cache-dir --user -r requirements.txt

# Unleash Start
### This is where we add things that we see it get stuck on, frequently, if our mission is to make the exploratory road easier
RUN apt-get update \
    && apt-get install -y build-essential curl apt-utils sudo git openssh-client sbcl

# ssh key sorts github out
RUN ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa

# if it wants to play with Rust, why not?
RUN curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y

# requirements-docker.txt is where to dump extra libraries that we get frequently stuck on
COPY --chown=appuser:appuser requirements-docker.txt .
RUN pip install --no-cache-dir --user -r requirements-docker.txt
# Unleash end

# Copy the application files
COPY --chown=appuser:appuser autogpt/ ./autogpt

# Set the entrypoint
ENTRYPOINT ["python", "-m", "autogpt"]
