#!/usr/bin/env bash
# Runs k6 on this machine in Docker, with this directory mounted, so Docker is
# the only thing to install. Arguments go to k6 unchanged:
#
#   ./k6.sh run tests/workload.js
#
# The image is the pinned grafana/k6 release plus the xk6-sse extension (see
# Dockerfile), which the copilot scenario needs to time a streamed turn. The
# first run builds it, in about two minutes; later runs reuse it. K6_IMAGE
# names another image to run instead, and K6_REBUILD_IMAGE=true rebuilds.
#
# Settings and secrets travel by name: each K6_*, LOAD_TEST_*, RUN_*,
# STRIPE_TEST_SECRET_KEY or test setting (PR_NUMBER, LOAD_PROFILE, SCENARIOS,
# ...) exported in the calling shell is passed into the container without its
# value appearing on a command line.
#
# Grafana Cloud k6 bills every virtual-user hour, so these tests run locally.
# A command that would use it (k6 cloud ..., --out cloud, K6_OUT=cloud) is
# refused unless --cloud comes first and K6_CLOUD_TOKEN is set:
#
#   ./k6.sh --cloud cloud run tests/workload.js
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

# The image tag names the k6 release and the extension version baked in, so
# bumping either in the Dockerfile builds a new image instead of reusing a
# stale one.
k6_version="$(sed -n 's/^ARG K6_VERSION=//p' Dockerfile)"
sse_version="$(sed -n 's/^ARG XK6_SSE_VERSION=//p' Dockerfile)"
default_image="autogpt-k6:${k6_version}-sse-${sse_version}"
image="${K6_IMAGE:-$default_image}"

if [ -z "${K6_IMAGE:-}" ]; then
  if [ "${K6_REBUILD_IMAGE:-}" = true ] ||
    ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "k6.sh: building $image (k6 ${k6_version} + xk6-sse ${sse_version}); this takes about two minutes the first time" >&2
    # A Docker config that points at a credential helper can stall the pull of
    # the public base images; an empty, temporary config has no helper and
    # leaves the caller's Docker login untouched.
    build_config="$(mktemp -d)"
    echo '{}' >"$build_config/config.json"
    DOCKER_CONFIG="$build_config" docker build --quiet -t "$image" . >/dev/null
    rm -rf "$build_config"
  fi
fi

# Each run gets an id unless the caller set one: it names the run's throwaway
# accounts and its summary file, results/<RUN_ID>.json.
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM}"
mkdir -p results

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
    K6_IMAGE | K6_REBUILD_IMAGE) ;;
    K6_* | LOAD_TEST_* | RUN_* | PR_NUMBER | LOAD_PROFILE | SCENARIOS | \
      RATE_SCALE | WARMUP | DURATION | COPILOT_* | GRAPH_* | SEED_* | \
      THRESHOLD_* | STRIPE_TEST_SECRET_KEY)
      docker_args+=(-e "$name")
      ;;
  esac
done

exec docker "${docker_args[@]}" "$image" "$@"
