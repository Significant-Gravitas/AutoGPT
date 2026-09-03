#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 || -z "$1" ]]; then
  echo "usage: $0 IMAGE" >&2
  exit 2
fi

readonly image="$1"
readonly max_attempts=3

for attempt in 1 2 3; do
  echo "Pulling ${image} (attempt ${attempt}/${max_attempts})"
  if docker pull "$image"; then
    exit 0
  fi

  if [[ "$attempt" -eq "$max_attempts" ]]; then
    echo "Failed to pull ${image} after ${max_attempts} attempts" >&2
    exit 1
  fi

  delay_seconds=$((attempt * 2))
  echo "Retrying ${image} in ${delay_seconds}s" >&2
  sleep "$delay_seconds"
done
