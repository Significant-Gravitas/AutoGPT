#!/bin/bash
#
# Database Migration Script: Supabase to GCP Cloud SQL
#
# This script migrates the AutoGPT Platform database from Supabase to a new PostgreSQL instance.
#
# Migration Steps:
#   0. Nuke destination database (drop schema, recreate, apply migrations)
#   1. Export platform schema data from source
#   2. Export auth.users data from source (for password hashes, OAuth IDs)
#   3. Import platform schema data to destination
#   4. Update User table in destination with auth data
#   5. Refresh materialized views
#
# Prerequisites:
#   - pg_dump and psql (PostgreSQL 15+)
#   - poetry installed (for Prisma migrations)
#   - Source and destination databases accessible
#
# Usage:
#   ./scripts/migrate_to_gcp.sh \
#     --source 'postgresql://user:pass@host:5432/db?schema=platform' \
#     --dest 'postgresql://user:pass@host:5432/db?schema=platform'
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${BACKEND_DIR}/migration_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Command line arguments
SOURCE_URL=""
DEST_URL=""
DRY_RUN=false

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    cat << EOF
Usage: $(basename "$0") --source <url> --dest <url> [options]

Required:
  --source <url>    Source database URL with ?schema=platform
  --dest <url>      Destination database URL with ?schema=platform

Options:
  --dry-run         Preview without making changes
  --help            Show this help

Migration Steps:
  0. Nuke destination database (DROP SCHEMA, recreate, apply Prisma migrations)
  1. Export platform schema data from source (READ-ONLY)
  2. Export auth.users data from source (READ-ONLY)
  3. Import platform data to destination
  4. Update User table with auth data (passwords, OAuth IDs)
  5. Refresh materialized views

EOF
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --source) SOURCE_URL="$2"; shift 2 ;;
            --dest) DEST_URL="$2"; shift 2 ;;
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

test_connections() {
    local source_base=$(get_base_url "$SOURCE_URL")
    local dest_base=$(get_base_url "$DEST_URL")

    log_info "Testing source connection..."
    if ! psql "${source_base}" -c "SELECT 1" > /dev/null 2>&1; then
        log_error "Cannot connect to source database"
        psql "${source_base}" -c "SELECT 1" 2>&1 || true
        exit 1
    fi
    log_success "Source connection OK"

    log_info "Testing destination connection..."
    if ! psql "${dest_base}" -c "SELECT 1" > /dev/null 2>&1; then
        log_error "Cannot connect to destination database"
        psql "${dest_base}" -c "SELECT 1" 2>&1 || true
        exit 1
    fi
    log_success "Destination connection OK"
}

# ============================================
# STEP 0: Nuke destination database
# ============================================
nuke_destination() {
    local schema=$(get_schema_from_url "$DEST_URL")
    local dest_base=$(get_base_url "$DEST_URL")

    log_info "=== STEP 0: Nuking destination database ==="

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would drop and recreate schema '${schema}' in destination"
        return
    fi

    # Show what exists in destination
    log_info "Current destination state:"
    local user_count=$(psql "${dest_base}" -t -c "SELECT COUNT(*) FROM ${schema}.\"User\"" 2>/dev/null | tr -d ' ' || echo "0")
    local graph_count=$(psql "${dest_base}" -t -c "SELECT COUNT(*) FROM ${schema}.\"AgentGraph\"" 2>/dev/null | tr -d ' ' || echo "0")
    echo "  - Users: ${user_count}"
    echo "  - AgentGraphs: ${graph_count}"

    echo ""
    log_warn "⚠️  WARNING: This will PERMANENTLY DELETE all data in the destination database!"
    log_warn "Schema '${schema}' will be dropped and recreated."
    echo ""
    read -p "Type 'NUKE' to confirm deletion: " -r
    echo ""

    if [[ "$REPLY" != "NUKE" ]]; then
        log_info "Cancelled - destination not modified"
        exit 0
    fi

    log_info "Dropping schema '${schema}'..."
    psql "${dest_base}" -c "DROP SCHEMA IF EXISTS ${schema} CASCADE;"

    log_info "Recreating schema '${schema}'..."
    psql "${dest_base}" -c "CREATE SCHEMA ${schema};"

    log_info "Applying Prisma migrations..."
    cd "${BACKEND_DIR}"
    DATABASE_URL="${DEST_URL}" DIRECT_URL="${DEST_URL}" poetry run prisma migrate deploy

    log_success "Destination database reset complete"
}

# ============================================
# STEP 1: Export platform schema data
# ============================================
export_platform_data() {
    local schema=$(get_schema_from_url "$SOURCE_URL")
    local base_url=$(get_base_url "$SOURCE_URL")
    local output_file="${BACKUP_DIR}/platform_data_${TIMESTAMP}.sql"

    log_info "=== STEP 1: Exporting platform schema data ==="
    mkdir -p "${BACKUP_DIR}"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would export schema '${schema}' to ${output_file}"
        log_info "DRY RUN: Excluding large execution tables"
        touch "$output_file"
        echo "$output_file"
        return
    fi

    log_info "Exporting from schema: ${schema}"
    log_info "EXCLUDING: AgentGraphExecution, AgentNodeExecution, AgentNodeExecutionInputOutput, AgentNodeExecutionKeyValueData, NotificationEvent"

    pg_dump "${base_url}" \
        --schema="${schema}" \
        --format=plain \
        --no-owner \
        --no-privileges \
        --data-only \
        --exclude-table="${schema}.AgentGraphExecution" \
        --exclude-table="${schema}.AgentNodeExecution" \
        --exclude-table="${schema}.AgentNodeExecutionInputOutput" \
        --exclude-table="${schema}.AgentNodeExecutionKeyValueData" \
        --exclude-table="${schema}.NotificationEvent" \
        --file="${output_file}" 2>&1

    # Remove Supabase-specific commands that break import
    sed -i.bak '/\\restrict/d' "${output_file}"
    rm -f "${output_file}.bak"

    local size=$(du -h "${output_file}" | cut -f1)
    log_success "Platform data exported: ${output_file} (${size})"
    echo "$output_file"
}

# ============================================
# STEP 2: Export auth.users data
# ============================================
export_auth_data() {
    local base_url=$(get_base_url "$SOURCE_URL")
    local output_file="${BACKUP_DIR}/auth_users_${TIMESTAMP}.csv"

    log_info "=== STEP 2: Exporting auth.users data ==="

    # Check if auth.users exists
    local auth_exists=$(psql "${base_url}" -t -c "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'auth' AND table_name = 'users')" 2>/dev/null | tr -d ' ')

    if [[ "$auth_exists" != "t" ]]; then
        log_warn "No auth.users table found - skipping auth export"
        echo ""
        return
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would export auth.users to ${output_file}"
        touch "$output_file"
        echo "$output_file"
        return
    fi

    log_info "Extracting auth data (passwords, OAuth IDs, email verification)..."

    psql "${base_url}" -c "\COPY (
        SELECT
            id,
            encrypted_password,
            (email_confirmed_at IS NOT NULL) as email_verified,
            CASE
                WHEN raw_app_meta_data->>'provider' = 'google'
                THEN raw_app_meta_data->>'provider_id'
                ELSE NULL
            END as google_id
        FROM auth.users
        WHERE encrypted_password IS NOT NULL
           OR raw_app_meta_data->>'provider' = 'google'
    ) TO '${output_file}' WITH CSV HEADER"

    local count=$(wc -l < "${output_file}" | tr -d ' ')
    log_success "Auth data exported: ${output_file} (${count} rows including header)"
    echo "$output_file"
}

# ============================================
# STEP 3: Import platform data to destination
# ============================================
import_platform_data() {
    local platform_file="$1"
    local dest_base=$(get_base_url "$DEST_URL")

    log_info "=== STEP 3: Importing platform data to destination ==="

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would import ${platform_file} to destination"
        return
    fi

    if [[ ! -f "$platform_file" ]]; then
        log_error "Platform data file not found: ${platform_file}"
        exit 1
    fi

    log_info "Importing platform data (this may take a while)..."

    # Import with error logging
    psql "${dest_base}" -f "${platform_file}" 2>&1 | tee "${BACKUP_DIR}/import_log_${TIMESTAMP}.txt" | head -100

    log_success "Platform data import completed"
}

# ============================================
# STEP 4: Update User table with auth data
# ============================================
update_user_auth_data() {
    local auth_file="$1"
    local schema=$(get_schema_from_url "$DEST_URL")
    local dest_base=$(get_base_url "$DEST_URL")

    log_info "=== STEP 4: Updating User table with auth data ==="

    if [[ -z "$auth_file" || ! -f "$auth_file" ]]; then
        log_warn "No auth data file - skipping User auth update"
        return
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would update User table with auth data"
        return
    fi

    log_info "Creating temporary table for auth data..."

    psql "${dest_base}" << EOF
-- Create temp table for auth data
CREATE TEMP TABLE temp_auth_users (
    id UUID,
    encrypted_password TEXT,
    email_verified BOOLEAN,
    google_id TEXT
);

-- Import CSV
\COPY temp_auth_users FROM '${auth_file}' WITH CSV HEADER;

-- Update User table with password hashes
UPDATE ${schema}."User" u
SET "passwordHash" = t.encrypted_password
FROM temp_auth_users t
WHERE u.id = t.id
AND t.encrypted_password IS NOT NULL
AND u."passwordHash" IS NULL;

-- Update User table with email verification
UPDATE ${schema}."User" u
SET "emailVerified" = t.email_verified
FROM temp_auth_users t
WHERE u.id = t.id
AND t.email_verified = true;

-- Update User table with Google OAuth IDs
UPDATE ${schema}."User" u
SET "googleId" = t.google_id
FROM temp_auth_users t
WHERE u.id = t.id
AND t.google_id IS NOT NULL
AND u."googleId" IS NULL;

-- Show results
SELECT
    'Total Users' as metric, COUNT(*)::text as value FROM ${schema}."User"
UNION ALL
SELECT 'With Password', COUNT(*)::text FROM ${schema}."User" WHERE "passwordHash" IS NOT NULL
UNION ALL
SELECT 'With Google OAuth', COUNT(*)::text FROM ${schema}."User" WHERE "googleId" IS NOT NULL
UNION ALL
SELECT 'Email Verified', COUNT(*)::text FROM ${schema}."User" WHERE "emailVerified" = true;

DROP TABLE temp_auth_users;
EOF

    log_success "User auth data updated"
}

# ============================================
# STEP 5: Refresh materialized views
# ============================================
refresh_views() {
    local schema=$(get_schema_from_url "$DEST_URL")
    local dest_base=$(get_base_url "$DEST_URL")

    log_info "=== STEP 5: Refreshing materialized views ==="

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would refresh materialized views"
        return
    fi

    psql "${dest_base}" << EOF
SET search_path TO ${schema};
REFRESH MATERIALIZED VIEW "mv_agent_run_counts";
REFRESH MATERIALIZED VIEW "mv_review_stats";

-- Reset sequences
SELECT setval(
    pg_get_serial_sequence('${schema}."SearchTerms"', 'id'),
    COALESCE((SELECT MAX(id) FROM ${schema}."SearchTerms"), 0) + 1,
    false
);
EOF

    log_success "Materialized views refreshed"
}

# ============================================
# Verification
# ============================================
verify_migration() {
    local source_base=$(get_base_url "$SOURCE_URL")
    local dest_base=$(get_base_url "$DEST_URL")
    local schema=$(get_schema_from_url "$SOURCE_URL")

    log_info "=== VERIFICATION ==="

    echo ""
    echo "Source counts:"
    psql "${source_base}" -c "SELECT 'User' as table_name, COUNT(*) FROM ${schema}.\"User\" UNION ALL SELECT 'AgentGraph', COUNT(*) FROM ${schema}.\"AgentGraph\" UNION ALL SELECT 'Profile', COUNT(*) FROM ${schema}.\"Profile\""

    echo ""
    echo "Destination counts:"
    psql "${dest_base}" -c "SELECT 'User' as table_name, COUNT(*) FROM ${schema}.\"User\" UNION ALL SELECT 'AgentGraph', COUNT(*) FROM ${schema}.\"AgentGraph\" UNION ALL SELECT 'Profile', COUNT(*) FROM ${schema}.\"Profile\""
}

# ============================================
# Main
# ============================================
main() {
    echo ""
    echo "========================================"
    echo "  Database Migration Script"
    echo "========================================"
    echo ""

    parse_args "$@"

    log_info "Source: $(get_base_url "$SOURCE_URL")"
    log_info "Destination: $(get_base_url "$DEST_URL")"
    [[ "$DRY_RUN" == true ]] && log_warn "DRY RUN MODE"
    echo ""

    test_connections

    echo ""

    # Step 0: Nuke destination database (with confirmation)
    nuke_destination
    echo ""

    if [[ "$DRY_RUN" != true ]]; then
        log_warn "This will migrate data to the destination database."
        read -p "Continue with migration? (y/N) " -n 1 -r
        echo ""
        [[ ! $REPLY =~ ^[Yy]$ ]] && { log_info "Cancelled"; exit 0; }
    fi

    echo ""
    log_info "Starting migration at $(date)"
    echo ""

    # Step 1: Export platform data (READ-ONLY on source)
    platform_file=$(export_platform_data)
    echo ""

    # Step 2: Export auth data (READ-ONLY on source)
    auth_file=$(export_auth_data)
    echo ""

    # Step 3: Import platform data to destination
    import_platform_data "$platform_file"
    echo ""

    # Step 4: Update User table with auth data
    update_user_auth_data "$auth_file"
    echo ""

    # Step 5: Refresh materialized views
    refresh_views
    echo ""

    # Verification
    verify_migration

    echo ""
    log_success "Migration completed at $(date)"
    echo ""
    echo "Files created:"
    echo "  - Platform data: ${platform_file}"
    [[ -n "$auth_file" ]] && echo "  - Auth data: ${auth_file}"
    echo ""
}

main "$@"
