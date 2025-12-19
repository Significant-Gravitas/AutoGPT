#!/bin/bash
#
# Database Migration Script: Supabase to GCP Cloud SQL
#
# This script automates the migration of the AutoGPT Platform database
# from a source PostgreSQL database to a destination PostgreSQL database.
#
# Prerequisites:
#   - pg_dump and psql (PostgreSQL 15+)
#   - poetry installed for Python scripts
#
# Usage:
#   cd backend
#   chmod +x scripts/migrate_to_gcp.sh
#   ./scripts/migrate_to_gcp.sh \
#     --source "postgresql://user:pass@host:5432/db?schema=platform" \
#     --dest "postgresql://user:pass@host:5433/db?schema=platform"
#
# Examples:
#   # Full migration
#   ./scripts/migrate_to_gcp.sh \
#     --source "postgresql://postgres.xxx:password@aws-0-us-east-1.pooler.supabase.com:5432/postgres?schema=platform" \
#     --dest "postgresql://postgres:password@127.0.0.1:5433/postgres?schema=platform"
#
#   # Dry run (preview only)
#   ./scripts/migrate_to_gcp.sh \
#     --source "postgresql://..." \
#     --dest "postgresql://..." \
#     --dry-run
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
BACKUP_DIR="${BACKEND_DIR}/migration_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Command line arguments
SOURCE_URL=""
DEST_URL=""
DRY_RUN=false
SKIP_AUTH_MIGRATION=false

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage help
usage() {
    cat << EOF
Usage: $(basename "$0") --source <url> --dest <url> [options]

Required:
  --source <url>       Source database URL (e.g., postgresql://user:pass@host:5432/db?schema=platform)
  --dest <url>         Destination database URL (e.g., postgresql://user:pass@host:5433/db?schema=platform)

Options:
  --dry-run            Preview migration without making changes
  --skip-auth          Skip the Supabase auth migration step (if already done)
  --help               Show this help message

Examples:
  # Migrate from Supabase to GCP
  $(basename "$0") \\
    --source "postgresql://postgres.xxx:pass@supabase.com:5432/postgres?schema=platform" \\
    --dest "postgresql://postgres:pass@127.0.0.1:5433/postgres?schema=platform"

  # Dry run to preview
  $(basename "$0") --source "..." --dest "..." --dry-run

EOF
    exit 1
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --source)
                SOURCE_URL="$2"
                shift 2
                ;;
            --dest)
                DEST_URL="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-auth)
                SKIP_AUTH_MIGRATION=true
                shift
                ;;
            --help|-h)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$SOURCE_URL" ]]; then
        log_error "Missing required argument: --source"
        usage
    fi

    if [[ -z "$DEST_URL" ]]; then
        log_error "Missing required argument: --dest"
        usage
    fi
}

# Extract schema from URL (default to 'platform' if not specified)
get_schema_from_url() {
    local url="$1"
    local schema=$(echo "$url" | grep -oP 'schema=\K[^&]+' || echo "platform")
    echo "$schema"
}

# Get URL without query params (for pg_dump/pg_restore)
get_base_url() {
    local url="$1"
    echo "${url%%\?*}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing=()

    if ! command -v pg_dump &> /dev/null; then
        missing+=("pg_dump")
    fi

    if ! command -v psql &> /dev/null; then
        missing+=("psql")
    fi

    if ! command -v poetry &> /dev/null; then
        missing+=("poetry")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Please install them before continuing."
        exit 1
    fi

    # Check PostgreSQL version
    local pg_version=$(pg_dump --version | grep -oE '[0-9]+' | head -1)
    if [ "$pg_version" -lt 15 ]; then
        log_warn "PostgreSQL version $pg_version detected. Version 15+ recommended."
    fi

    log_success "All prerequisites satisfied"
}

# Test database connections
test_connections() {
    log_info "Testing source database connection..."
    if ! psql "${SOURCE_URL}" -c "SELECT 1" &> /dev/null; then
        log_error "Cannot connect to source database"
        log_error "URL: ${SOURCE_URL%%:*}:****@${SOURCE_URL#*@}"
        exit 1
    fi
    log_success "Source database connection OK"

    log_info "Testing destination database connection..."
    if ! psql "${DEST_URL}" -c "SELECT 1" &> /dev/null; then
        log_error "Cannot connect to destination database"
        log_error "URL: ${DEST_URL%%:*}:****@${DEST_URL#*@}"
        exit 1
    fi
    log_success "Destination database connection OK"
}

# Run auth migration on source database
run_auth_migration() {
    if [[ "$SKIP_AUTH_MIGRATION" == true ]]; then
        log_info "Skipping auth migration (--skip-auth flag set)"
        return
    fi

    log_info "Running auth migration on source database..."

    cd "${BACKEND_DIR}"

    # Check if auth.users table exists
    local auth_exists=$(psql "${SOURCE_URL}" -t -c "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'auth' AND table_name = 'users')" 2>/dev/null | tr -d ' ')

    if [[ "$auth_exists" != "t" ]]; then
        log_warn "No auth.users table found in source database"
        log_warn "Skipping Supabase auth migration (may not be a Supabase database)"
        return
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would run auth migration script"
        DATABASE_URL="${SOURCE_URL}" \
        DIRECT_URL="${SOURCE_URL}" \
        poetry run python scripts/migrate_supabase_users.py --dry-run
    else
        # Apply Prisma migrations first
        log_info "Applying Prisma migrations to source..."
        DATABASE_URL="${SOURCE_URL}" \
        DIRECT_URL="${SOURCE_URL}" \
        poetry run prisma migrate deploy

        # Run the Supabase user migration script
        log_info "Migrating auth data from Supabase auth.users..."
        DATABASE_URL="${SOURCE_URL}" \
        DIRECT_URL="${SOURCE_URL}" \
        poetry run python scripts/migrate_supabase_users.py
    fi

    log_success "Auth migration completed"
}

# Export database from source
export_database() {
    local schema=$(get_schema_from_url "$SOURCE_URL")
    local base_url=$(get_base_url "$SOURCE_URL")
    local backup_file="${BACKUP_DIR}/platform_backup_${TIMESTAMP}.dump"

    log_info "Exporting database from source (schema: ${schema})..."

    mkdir -p "${BACKUP_DIR}"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would export schema '${schema}' to ${backup_file}"
        # Create empty file for dry run
        touch "${backup_file}"
        echo "$backup_file"
        return
    fi

    log_info "This may take a while depending on database size..."

    pg_dump "${base_url}" \
        --schema="${schema}" \
        --format=custom \
        --no-owner \
        --no-privileges \
        --verbose \
        --file="${backup_file}" 2>&1 | while read line; do
            echo -e "${BLUE}[pg_dump]${NC} $line"
        done

    local file_size=$(du -h "${backup_file}" | cut -f1)
    log_success "Database exported to ${backup_file} (${file_size})"

    echo "$backup_file"
}

# Set up schema on destination
setup_dest_schema() {
    log_info "Setting up schema on destination database..."

    cd "${BACKEND_DIR}"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would deploy Prisma schema to destination"
        return
    fi

    # Deploy Prisma schema to destination
    DATABASE_URL="${DEST_URL}" \
    DIRECT_URL="${DEST_URL}" \
    poetry run prisma migrate deploy

    log_success "Destination schema created via Prisma migrations"
}

# Import data to destination
import_database() {
    local backup_file="$1"
    local schema=$(get_schema_from_url "$DEST_URL")
    local base_url=$(get_base_url "$DEST_URL")

    log_info "Importing data to destination database..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would import ${backup_file} to destination"
        return
    fi

    log_info "This may take a while..."

    pg_restore \
        --dbname="${base_url}" \
        --schema="${schema}" \
        --no-owner \
        --no-privileges \
        --verbose \
        --data-only \
        --disable-triggers \
        "${backup_file}" 2>&1 | while read line; do
            echo -e "${BLUE}[pg_restore]${NC} $line"
        done || true  # pg_restore returns non-zero on warnings

    log_success "Data import completed"
}

# Refresh materialized views
refresh_views() {
    local schema=$(get_schema_from_url "$DEST_URL")

    log_info "Refreshing materialized views..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would refresh materialized views"
        return
    fi

    psql "${DEST_URL}" <<EOF
SET search_path TO ${schema};

-- Refresh materialized views
REFRESH MATERIALIZED VIEW "mv_agent_run_counts";
REFRESH MATERIALIZED VIEW "mv_review_stats";

-- Reset sequences
SELECT setval(
    pg_get_serial_sequence('${schema}."SearchTerms"', 'id'),
    COALESCE((SELECT MAX(id) FROM "${schema}"."SearchTerms"), 0) + 1,
    false
);
EOF

    log_success "Materialized views refreshed"
}

# Verify migration
verify_migration() {
    log_info "Verifying migration..."

    echo ""
    echo "====== SOURCE DATABASE ROW COUNTS ======"
    psql "${SOURCE_URL}" -f "${SCRIPT_DIR}/verify_migration.sql"

    echo ""
    echo "====== DESTINATION DATABASE ROW COUNTS ======"
    psql "${DEST_URL}" -f "${SCRIPT_DIR}/verify_migration.sql"

    echo ""
    log_info "Compare the counts above to verify migration success"
}

# Print summary
print_summary() {
    local backup_file="$1"

    echo ""
    echo "========================================"
    echo "         MIGRATION COMPLETE"
    echo "========================================"
    echo ""
    log_success "Database has been migrated successfully"
    echo ""
    echo "Backup file: ${backup_file}"
    echo ""
    echo "Next steps:"
    echo "  1. Verify the row counts above match"
    echo "  2. Update your application's DATABASE_URL to point to destination"
    echo "  3. Restart your application services"
    echo "  4. Test login with existing user credentials"
    echo "  5. Keep source database available for 48 hours as backup"
    echo ""
}

# Main execution
main() {
    echo ""
    echo "========================================"
    echo "     Database Migration Script"
    echo "========================================"
    echo ""

    parse_args "$@"

    # Mask passwords in output
    local source_display="${SOURCE_URL%%:*}://****@${SOURCE_URL#*@}"
    local dest_display="${DEST_URL%%:*}://****@${DEST_URL#*@}"

    log_info "Source: ${source_display}"
    log_info "Destination: ${dest_display}"

    if [[ "$DRY_RUN" == true ]]; then
        log_warn "DRY RUN MODE - No changes will be made"
    fi
    echo ""

    check_prerequisites
    test_connections

    if [[ "$DRY_RUN" != true ]]; then
        echo ""
        log_warn "This will perform a database migration."
        log_warn "Ensure application services are stopped to prevent data loss."
        echo ""
        read -p "Continue with migration? (y/N) " -n 1 -r
        echo ""

        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Migration cancelled"
            exit 0
        fi
    fi

    echo ""
    log_info "Starting migration at $(date)"
    echo ""

    # Step 1: Auth migration on source
    run_auth_migration

    # Step 2: Export from source
    backup_file=$(export_database)

    # Step 3: Set up destination schema
    setup_dest_schema

    # Step 4: Import to destination
    import_database "$backup_file"

    # Step 5: Post-import tasks
    refresh_views

    # Step 6: Verify
    verify_migration

    # Summary
    if [[ "$DRY_RUN" != true ]]; then
        print_summary "$backup_file"
    else
        echo ""
        log_info "DRY RUN COMPLETE - No changes were made"
        echo ""
    fi

    log_info "Finished at $(date)"
}

# Run main function
main "$@"
