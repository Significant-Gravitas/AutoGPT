#!/usr/bin/env bash

# Script to quickly rebuild and run the app in a Docker container.
# This file is run manually and its image is used by execute_python_file command

docker rmi fsamir/nodejs-puppeteer:dev --force

docker build --build-arg base_image=seleniarm/standalone-chromium:103.0 \
  -t fsamir/nodejs-puppeteer:dev \
  -f pupeteer.Dockerfile \
  .


docker rm -f puppeteer || true
#Test Image
#https://pptr.dev/guides/docker
docker run -it \
           --memory=1200M \
           --shm-size 1G \
           --name puppeteer \
           -v /$(pwd)/auto_gpt_workspace/:/worspace \
           fsamir/nodejs-puppeteer:dev \
           bash -i -c 'npm run test'
#           node books-scraper.js
#            npm run test
#           /bin/bash
#           bash -i -c 'node books-scraper.js'

