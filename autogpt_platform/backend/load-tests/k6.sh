#!/usr/bin/env bash
# Runs k6 from the grafana/k6 image with this directory mounted, so Docker is
# the only thing to install. Arguments go to k6 unchanged:
#
#   ./k6.sh run tests/api/core-api-test.js
#
# Settings and secrets travel by name: each K6_*, LOAD_TEST_* or test setting
# (PR_NUMBER, VUS, DURATION, ...) exported in the calling shell is passed into
# the container without its value appearing on a command line.
set -euo pipefail

cd "$(dirname "$0")"

image="${K6_IMAGE:-grafana/k6:2.3.0}"
docker_args=(
  run --rm -i
  --user "$(id -u):$(id -g)"
  -v "$PWD:/load-tests" -w /load-tests
  # K6_ENVIRONMENT=LOCAL reaches the stack on the host through this name.
  --add-host=host.docker.internal:host-gateway
  -e "LOCAL_HOST=${LOCAL_HOST:-host.docker.internal}"
)
if [ -t 0 ] && [ -t 1 ]; then
  docker_args+=(-t)
fi

for name in $(compgen -e); do
  case "$name" in
    K6_* | LOAD_TEST_* | PR_NUMBER | VUS | DURATION | RAMP_UP | RAMP_DOWN | \
      REQUESTS_PER_VU | CONCURRENT_REQUESTS | ENDPOINT | THRESHOLD_*)
      docker_args+=(-e "$name")
      ;;
  esac
done

exec docker "${docker_args[@]}" "$image" "$@"
