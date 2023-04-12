#FROM node:16
#
#RUN apt-get update \
# && apt-get install -y chromium \
#    fonts-ipafont-gothic fonts-wqy-zenhei fonts-thai-tlwg fonts-kacst fonts-freefont-ttf libxss1
##    \
##    --no-install-recommends
#
#
## non-root user that comes with `node` images.
#USER node
##RUN mkdir -p /home/node

#Not arm compatible:
ARG base_image=base_image=seleniarm/standalone-chromium:103.0
#FROM --platform=linux/arm64/v7 $base_image
FROM $base_image

ENV CHROMEDRIVER_PORT 4444
ENV CHROMEDRIVER_WHITELISTED_IPS "127.0.0.1"
ENV CHROMEDRIVER_URL_BASE ''
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
ENV PUPPETEER_PRODUCT=chrome
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium

#Force all commans to use bash - nedded for nvm
SHELL ["/bin/bash", "-i", "-c"]

ENV NODE_VERSION=16.13.0
RUN wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash

USER seluser

WORKDIR /app

COPY --chown=seluser package.json .
#COPY --chown=seluser package-lock.json .
RUN npm install

COPY --chown=seluser books-scraper.js .
COPY --chown=seluser books-scraper-test.js .


