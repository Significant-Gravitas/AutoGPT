#!/bin/bash
source .env

python ./scripts/check_requirements.py requirements.txt
if [ $? -eq 1 ]
then
    echo Installing missing packages...
    pip install -r requirements.txt
fi
if [ "$DEPENDS_ON" == "local" ]; then
  docker-compose -f docker-compose.local.yml up -d --remove-orphans
elif [ "$DEPENDS_ON" == "redis" ]; then
  docker-compose -f docker-compose.redis.yml up -d --remove-orphans
elif [ "$DEPENDS_ON" == "weaviate" ]; then
  docker-compose -f docker-compose.weaviate.yml up -d --remove-orphans
else
  echo Error: Unsupported value for DEPENDS_ON in .env file. Running Default configuration
  docker-compose up -d
fi
python -m autogpt $@
read -p "Press any key to continue..."
echo "Stopping all running containers..."
for container_id in $(docker ps -q); do docker stop -t 10 $container_id; done

