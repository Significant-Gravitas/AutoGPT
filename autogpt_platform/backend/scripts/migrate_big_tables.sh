#!/bin/bash
#
# Migrate Large Tables: Stream execution history from source to destination
#
# This script streams the large execution tables that were excluded from
# the initial migration. Run this AFTER migrate_to_gcp.sh completes.
#
# Tables migrated (in order of size):
#   - NotificationEvent (94 MB)
#   - AgentNodeExecutionKeyValueData (792 KB)
#   - AgentGraphExecution (1.3 GB)
#   - AgentNodeExecution (6 GB)
#   - AgentNodeExecutionInputOutput (30 GB)
#
# Usage:
#   ./scripts/migrate_big_tables.sh \
#     --source 'postgresql://user:pass@host:5432/db?schema=platform' \
#     --dest 'postgresql://user:pass@host:5432/db?schema=platform'
#
# Options:
#   --table <name>    Migrate only a specific table
#   --dry-run         Show what would be done without migrating
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Arguments
SOURCE_URL=""
DEST_URL=""
DRY_RUN=false
SINGLE_TABLE=""

# Tables to migrate (ordered smallest to largest)
TABLES=(
    "NotificationEvent"
    "AgentNodeExecutionKeyValueData"
    "AgentGraphExecution"
    "AgentNodeExecution"
    "AgentNodeExecutionInputOutput"
)

usage() {
    cat << EOF
Usage: $(basename "$0") --source <url> --dest <url> [options]

Required:
  --source <url>    Source database URL with ?schema=platform
  --dest <url>      Destination database URL with ?schema=platform

Options:
  --table <name>    Migrate only a specific table (e.g., AgentGraphExecution)
  --dry-run         Show what would be done without migrating
  --help            Show this help

Tables migrated (in order):
  1. NotificationEvent (94 MB)
  2. AgentNodeExecutionKeyValueData (792 KB)
  3. AgentGraphExecution (1.3 GB)
  4. AgentNodeExecution (6 GB)
  5. AgentNodeExecutionInputOutput (30 GB)

EOF
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --source) SOURCE_URL="$2"; shift 2 ;;
            --dest) DEST_URL="$2"; shift 2 ;;
            --table) SINGLE_TABLE="$2"; shift 2 ;;
            --dry-run) DRY_RUN=true; shift ;;
            --help|-h) usage ;;
            *) log_error "Unknown option: $1"; usage ;;
        esac
    done

    if [[ -z "$SOURCE_URL" ]]; then
        log_error "Missing --source"
        usage
    fi

    if [[ -z "$DEST_URL" ]]; then
        log_error "Missing --dest"
        usage
    fi
}

get_schema_from_url() {
    local url="$1"
    local schema=$(echo "$url" | sed -n 's/.*schema=\([^&]*\).*/\1/p')
    echo "${schema:-platform}"
}

get_base_url() {
    local url="$1"
    echo "${url%%\?*}"
}

get_table_size() {
    local base_url="$1"
    local schema="$2"
    local table="$3"

    psql "${base_url}" -t -c "
        SELECT pg_size_pretty(pg_total_relation_size('${schema}.\"${table}\"'))
    " 2>/dev/null | tr -d ' ' || echo "unknown"
}

get_table_count() {
    local base_url="$1"
    local schema="$2"
    local table="$3"

    psql "${base_url}" -t -c "
        SELECT COUNT(*) FROM ${schema}.\"${table}\"
    " 2>/dev/null | tr -d ' ' || echo "0"
}

migrate_table() {
    local table="$1"
    local source_base=$(get_base_url "$SOURCE_URL")
    local dest_base=$(get_base_url "$DEST_URL")
    local schema=$(get_schema_from_url "$SOURCE_URL")

    log_info "=== Migrating ${table} ==="

    # Get source stats
    local size=$(get_table_size "$source_base" "$schema" "$table")
    local count=$(get_table_count "$source_base" "$schema" "$table")
    log_info "Source: ${count} rows (${size})"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would stream ${table} from source to destination"
        return
    fi

    # Check if destination already has data
    local dest_count=$(get_table_count "$dest_base" "$schema" "$table")
    if [[ "$dest_count" != "0" ]]; then
        log_warn "Destination already has ${dest_count} rows in ${table}"
        read -p "Continue and add more rows? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping ${table}"
            return
        fi
    fi

    log_info "Streaming ${table} (this may take a while for large tables)..."
    local start_time=$(date +%s)

    # Stream directly from source to destination
    pg_dump "${source_base}" \
        --table="${schema}.\"${table}\"" \
        --data-only \
        --no-owner \
        --no-privileges \
        2>/dev/null \
        | grep -v '\\restrict' \
        | psql "${dest_base}" -q

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Verify
    local new_dest_count=$(get_table_count "$dest_base" "$schema" "$table")
    log_success "${table}: ${new_dest_count} rows migrated in ${duration}s"
}

main() {
    echo ""
    echo "========================================"
    echo "  Migrate Large Tables"
    echo "========================================"
    echo ""

    parse_args "$@"

    local source_base=$(get_base_url "$SOURCE_URL")
    local dest_base=$(get_base_url "$DEST_URL")

    log_info "Source: ${source_base}"
    log_info "Destination: ${dest_base}"
    [[ "$DRY_RUN" == true ]] && log_warn "DRY RUN MODE"
    echo ""

    # Test connections
    log_info "Testing connections..."
    if ! psql "${source_base}" -c "SELECT 1" > /dev/null 2>&1; then
        log_error "Cannot connect to source"
        exit 1
    fi
    if ! psql "${dest_base}" -c "SELECT 1" > /dev/null 2>&1; then
        log_error "Cannot connect to destination"
        exit 1
    fi
    log_success "Connections OK"
    echo ""

    # Determine which tables to migrate
    local tables_to_migrate=()
    if [[ -n "$SINGLE_TABLE" ]]; then
        tables_to_migrate=("$SINGLE_TABLE")
    else
        tables_to_migrate=("${TABLES[@]}")
    fi

    # Show plan
    log_info "Tables to migrate:"
    local schema=$(get_schema_from_url "$SOURCE_URL")
    for table in "${tables_to_migrate[@]}"; do
        local size=$(get_table_size "$source_base" "$schema" "$table")
        echo "  - ${table} (${size})"
    done
    echo ""

    if [[ "$DRY_RUN" != true ]]; then
        log_warn "This will stream large amounts of data to the destination."
        read -p "Continue? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Cancelled"
            exit 0
        fi
    fi

    echo ""
    log_info "Starting migration at $(date)"
    echo ""

    # Migrate each table
    for table in "${tables_to_migrate[@]}"; do
        migrate_table "$table"
        echo ""
    done

    log_success "Migration completed at $(date)"
    echo ""
}

main "$@"
