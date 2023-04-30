services:
  app:
    entrypoint:
    - sleep
    - infinity
    image: docker/dev-environments-default:stable-1
    init: true
    volumes:
    - type: bind
      source: /var/run/docker.sock
      target: /var/run/docker.sock

