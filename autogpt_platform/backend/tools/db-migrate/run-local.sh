#!/bin/bash
#
# Build and run db-migrate locally against the databases
#
# Usage:
#   ./run-local.sh [command] [args...]
#
# Examples:
#   ./run-local.sh table-sizes
#   ./run-local.sh solo --user-id abc-123
#   ./run-local.sh quick --dry-run
#   ./run-local.sh verify
#
# Environment variables required:
#   SOURCE_URL - Source database URL (Supabase)
#   DEST_URL   - Destination database URL (GCP Cloud SQL)
#
# You can create a .env file in this directory with:
#   SOURCE_URL=postgresql://user:pass@host:5432/db?schema=platform
#   DEST_URL=postgresql://user:pass@host:5432/db?schema=platform
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env file if it exists
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Check required env vars
if [[ -z "${SOURCE_URL:-}" ]]; then
    echo "ERROR: SOURCE_URL environment variable is required"
    echo "Set it or create a .env file in this directory"
    exit 1
fi

if [[ -z "${DEST_URL:-}" ]]; then
    echo "ERROR: DEST_URL environment variable is required"
    echo "Set it or create a .env file in this directory"
    exit 1
fi

echo "=== Building db-migrate ==="
cd "$SCRIPT_DIR"
cargo build --release

echo ""
echo "=== Running ==="
echo "Source: ${SOURCE_URL%%@*}@..."
echo "Dest: ${DEST_URL%%@*}@..."
echo ""

./target/release/db-migrate "$@"
