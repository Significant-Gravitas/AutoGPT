#!/usr/bin/env bash
# Installs the Prisma CLI and its engines, the only downloads `prisma generate`
# makes, so a dropped connection is retried and a schema error is not.

set -euo pipefail

# A dropped connection, not an outage: three tries span 45 s.
readonly max_attempts=3

for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  echo "Preparing the Prisma CLI (attempt ${attempt}/${max_attempts})"
  if poetry run prisma --version; then
    exit 0
  fi

  if [[ "$attempt" -eq "$max_attempts" ]]; then
    echo "Failed to install the Prisma CLI after ${max_attempts} attempts" >&2
    exit 1
  fi

  delay_seconds=$((attempt * 15))
  echo "Retrying the Prisma CLI install in ${delay_seconds}s" >&2
  sleep "$delay_seconds"
done

exit 1
