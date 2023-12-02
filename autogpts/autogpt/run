#!/bin/sh

kill $(lsof -t -i :8000)

if [ ! -f .env ] && [ -z "$OPENAI_API_KEY" ]; then
  cp .env.example .env
  echo "Please add your api keys to the .env file." >&2
  # exit 1
fi
poetry run serve --debug
