#!/usr/bin/env bash
# Runs k6 on this machine from the grafana/k6 image, with this directory
# mounted, so Docker is the only thing to install. Arguments go to k6
# unchanged:
#
#   ./k6.sh run tests/api/core-api-test.js
#
# Settings and secrets travel by name: each K6_*, LOAD_TEST_* or test setting
# (PR_NUMBER, VUS, DURATION, ...) exported in the calling shell is passed into
# the container without its value appearing on a command line.
#
# Grafana Cloud k6 bills every virtual-user hour, so these tests run locally.
# A command that would use it (k6 cloud ..., --out cloud, K6_OUT=cloud) is
# refused unless --cloud comes first and K6_CLOUD_TOKEN is set:
#
#   ./k6.sh --cloud cloud run tests/api/core-api-test.js
set -euo pipefail

cd "$(dirname "$0")"

cloud_opt_in=false
if [ "${1:-}" = "--cloud" ]; then
  cloud_opt_in=true
  shift
fi

uses_cloud=false
if [ "${1:-}" = "cloud" ]; then
  uses_cloud=true
fi
case "${K6_OUT:-}" in
  *cloud*) uses_cloud=true ;;
esac
previous=""
for arg in "$@"; do
  case "$arg" in
    --out=*cloud* | -o=*cloud*) uses_cloud=true ;;
  esac
  if { [ "$previous" = "--out" ] || [ "$previous" = "-o" ]; } &&
    [[ "$arg" == *cloud* ]]; then
    uses_cloud=true
  fi
  previous="$arg"
done

if [ "$uses_cloud" = true ]; then
  if [ "$cloud_opt_in" != true ] || [ -z "${K6_CLOUD_TOKEN:-}" ]; then
    {
      echo "k6.sh: this command would use Grafana Cloud k6, which bills every virtual-user hour."
      echo "k6.sh: run it on this machine instead. Grafana Cloud is only for load one machine"
      echo "k6.sh: cannot generate, agreed with the team first: then put --cloud first and set"
      echo "k6.sh: K6_CLOUD_TOKEN (see README.md)."
    } >&2
    exit 2
  fi
  echo "k6.sh: using Grafana Cloud k6, billed per virtual-user hour (peak VUs x duration); local runs are the default." >&2
fi

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
    # Without --cloud, k6 in the container gets no Grafana Cloud credentials.
    K6_CLOUD_*)
      if [ "$cloud_opt_in" = true ]; then
        docker_args+=(-e "$name")
      fi
      ;;
    K6_* | LOAD_TEST_* | PR_NUMBER | VUS | DURATION | RAMP_UP | RAMP_DOWN | \
      REQUESTS_PER_VU | CONCURRENT_REQUESTS | ENDPOINT | THRESHOLD_*)
      docker_args+=(-e "$name")
      ;;
  esac
done

exec docker "${docker_args[@]}" "$image" "$@"
