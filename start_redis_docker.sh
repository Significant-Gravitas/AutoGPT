#!/bin/sh
printf ">>> \033[32mRunning\033[0m 'docker-compose -f docker_redis.yml up -d' \n"
printf ">>> \033[31mStop\033[0m the Redis-Stack docker container with 'docker stop redis-stack' \n"
sleep 2
docker-compose -f docker_redis.yml up -d
