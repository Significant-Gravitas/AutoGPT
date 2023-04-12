FROM python:3.11

# Officially supported by playwright:
#FROM mcr.microsoft.com/playwright/python:v1.30.0-focal
#FROM mcr.microsoft.com/playwright/python:v1.32.0-focal

RUN pip install playwright
RUN playwright install --with-deps chromium

WORKDIR /app
#COPY scripts/ /app
COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

