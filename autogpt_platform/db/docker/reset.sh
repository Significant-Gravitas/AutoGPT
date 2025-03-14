#!/bin/bash

echo "WARNING: This will remove all containers and container data, and will reset the .env file. This action cannot be undone!"
read -p "Are you sure you want to proceed? (y/N) " -n 1 -r
echo    # Move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Operation cancelled."
    exit 1
fi

echo "Stopping and removing all containers..."
docker compose -f docker-compose.yml -f ./dev/docker-compose.dev.yml down -v --remove-orphans

echo "Cleaning up bind-mounted directories..."
BIND_MOUNTS=(
  "./volumes/db/data"
)

for DIR in "${BIND_MOUNTS[@]}"; do
  if [ -d "$DIR" ]; then
    echo "Deleting $DIR..."
    rm -rf "$DIR"
  else
    echo "Directory $DIR does not exist. Skipping bind mount deletion step..."
  fi
done

echo "Resetting .env file..."
if [ -f ".env" ]; then
  echo "Removing existing .env file..."
  rm -f .env
else
  echo "No .env file found. Skipping .env removal step..."
fi

if [ -f ".env.example" ]; then
  echo "Copying .env.example to .env..."
  cp .env.example .env
else
  echo ".env.example file not found. Skipping .env reset step..."
fi

echo "Cleanup complete!"