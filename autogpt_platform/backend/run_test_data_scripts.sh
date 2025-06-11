#!/bin/bash
# Script to run test data creation and update scripts

set -e  # Exit on error

echo "=================================================="
echo "Running Test Data Scripts for AutoGPT Platform"
echo "=================================================="
echo ""

# Check if we're in the backend directory
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: This script must be run from the backend directory"
    echo "Please cd to autogpt_platform/backend first"
    exit 1
fi

# Check if docker compose is available
if ! command -v docker &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker or docker-compose not found"
    echo "Please install Docker to continue"
    exit 1
fi

# Use docker compose v2 if available, otherwise fall back to docker-compose
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo "1. Starting test database services..."
echo "-----------------------------------"
$DOCKER_COMPOSE -f docker-compose.test.yaml up -d

# Wait for services to be ready
echo ""
echo "2. Waiting for services to be healthy..."
echo "--------------------------------------"
sleep 5

# Check if postgres is ready
max_attempts=30
attempt=0
while ! $DOCKER_COMPOSE -f docker-compose.test.yaml exec -T postgres-test pg_isready -U postgres > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo "ERROR: PostgreSQL did not become ready in time"
        exit 1
    fi
    echo -n "."
    sleep 1
done
echo " PostgreSQL is ready!"

echo ""
echo "3. Running database migrations..."
echo "--------------------------------"
poetry run prisma migrate deploy

echo ""
echo "4. Running test data creator..."
echo "------------------------------"
cd test
poetry run python test_data_creator.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test data created successfully!"
    
    echo ""
    echo "5. Running test data updater..."
    echo "------------------------------"
    poetry run python test_data_updater.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Test data updated successfully!"
    else
        echo ""
        echo "❌ Test data updater failed!"
        exit 1
    fi
else
    echo ""
    echo "❌ Test data creator failed!"
    exit 1
fi

echo ""
echo "=================================================="
echo "Test data setup completed successfully!"
echo "=================================================="
echo ""
echo "The materialized views have been populated with test data."
echo "You can now run tests or use the application with sample data."
echo ""
echo "To stop the test database services, run:"
echo "  $DOCKER_COMPOSE -f docker-compose.test.yaml down"
echo ""