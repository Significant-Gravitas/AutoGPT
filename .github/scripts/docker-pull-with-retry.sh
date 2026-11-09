#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 || -z "$1" ]]; then
  echo "usage: $0 IMAGE" >&2
  exit 2
fi

readonly image="$1"
# Docker Hub has refused every pull for 20+ s at a time; five tries span 150 s.
readonly max_attempts=5

for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  echo "Pulling ${image} (attempt ${attempt}/${max_attempts})"
  if docker pull "$image"; then
    exit 0
  fi

  if [[ "$attempt" -eq "$max_attempts" ]]; then
    echo "Failed to pull ${image} after ${max_attempts} attempts" >&2
    exit 1
  fi

  delay_seconds=$((attempt * 15))
  echo "Retrying ${image} in ${delay_seconds}s" >&2
  sleep "$delay_seconds"
done

exit 1
