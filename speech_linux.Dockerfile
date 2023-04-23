# Use an official Python base image from the Docker Hub
FROM python:3.10-slim
#FROM ubuntu:22.04

# Install git
RUN apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install git chromium-driver

# Install Xvfb and other dependencies for headless browser testing
RUN apt-get update \
    && apt-get install -y wget gnupg2 libgtk-3-0 libdbus-glib-1-2 dbus-x11 xvfb ca-certificates

# Install Firefox / Chromium
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y chromium firefox-esr \
    pulseaudio pulseaudio-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev  \
    libgstreamer-plugins-bad1.0-dev  \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly  \
    gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3  \
    gstreamer1.0-qt5 gstreamer1.0-pulseaudio jackd1

# Set environment variables
ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create a non-root user and set permissions
RUN useradd --create-home appuser
WORKDIR /home/appuser
RUN chown appuser:appuser /home/appuser


RUN DEBIAN_FRONTEND=noninteractive apt-get install -y


# Switch back to the appuser user
USER appuser

# Copy the requirements.txt file and install the requirements
COPY --chown=appuser:appuser requirements.txt .
RUN sed -i '/Items below this point will not be included in the Docker Image/,$d' requirements.txt && \
	pip3 install --no-cache-dir --user -r requirements.txt

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libcairo2-dev libgirepository1.0-dev
RUN pip3 install PyGObject

USER appuser
# Copy the application files
COPY --chown=appuser:appuser ./autogpt ./autogpt

# Set the entrypoint
ENTRYPOINT ["python3", "-m", "autogpt"]
