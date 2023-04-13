FROM python:3.10-slim

RUN apt-get update
RUN apt-get install -y git chromium-driver wget gnupg2 libgtk-3-0 libdbus-glib-1-2 dbus-x11 xvfb ca-certificates

RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y chromium firefox-esr

ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /tmp
COPY requirements.txt .

# Chop out the part of requirements.txt not intended for the docker image
RUN sed -i '/Items below this point will not be included in the Docker Image/,$d' requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app

COPY . .

ENTRYPOINT ["python", "-m", "autogpt"]
