#!/usr/bin/env python3
"""
AutoGPT Analytics — View Generator
====================================
Reads every .sql file in queries/ and registers it as a
CREATE OR REPLACE VIEW in the analytics schema.

Usage
-----
  # Print one-time setup SQL (schema, role, grants)
  python generate_views.py --setup

  # Dry-run: print all view SQL without executing
  python generate_views.py --dry-run

  # Apply to database (uses DATABASE_URL env var)
  DATABASE_URL="postgresql://analytics_readonly:pass@host:5432/postgres" \\
    python generate_views.py

  # Apply to database (explicit connection string)
  python generate_views.py --db-url "postgresql://..."

  # Apply only specific views
  python generate_views.py --only graph_execution,retention_login_weekly

Environment variables
---------------------
  DATABASE_URL   Postgres connection string (alternative to --db-url)

Notes
-----
- Run --setup output first (once) to create the schema and grants.
- Safe to re-run: uses CREATE OR REPLACE VIEW.
- Looker, PostHog Data Warehouse, and Supabase MCP all benefit
  from the same analytics.* views after running this script.
"""

import argparse
import os
import sys
from pathlib import Path

QUERIES_DIR = Path(__file__).parent / "queries"
SCHEMA = "analytics"

SETUP_SQL = """\
-- =============================================================
-- AutoGPT Analytics Schema Setup
-- Run ONCE in Supabase SQL Editor as the postgres superuser.
-- After this, run generate_views.py to create/refresh the views.
-- =============================================================

-- 1. Create the analytics schema
CREATE SCHEMA IF NOT EXISTS analytics;

-- 2. Create the read-only role (skip if already exists)
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'analytics_readonly') THEN
    CREATE ROLE analytics_readonly WITH LOGIN PASSWORD 'CHANGE_ME';
  END IF;
END
$$;

-- 3. Auth schema grants
--    Supabase restricts the auth schema; run as postgres superuser.
GRANT USAGE ON SCHEMA auth TO analytics_readonly;
GRANT SELECT ON auth.sessions TO analytics_readonly;
GRANT SELECT ON auth.audit_log_entries TO analytics_readonly;

-- 4. Platform schema grants
GRANT USAGE ON SCHEMA platform TO analytics_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA platform TO analytics_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA platform
  GRANT SELECT ON TABLES TO analytics_readonly;

-- 5. Analytics schema grants
GRANT USAGE ON SCHEMA analytics TO analytics_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO analytics_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics
  GRANT SELECT ON TABLES TO analytics_readonly;
"""


def view_name_from_file(path: Path) -> str:
    return path.stem


def build_view_sql(name: str, query_body: str) -> str:
    # Strip any trailing semicolons so we can wrap cleanly
    body = query_body.strip().rstrip(";")
    return f"CREATE OR REPLACE VIEW {SCHEMA}.{name} AS\n{body};\n"


def generate_all(only: list[str] | None = None) -> list[tuple[str, str]]:
    """Return list of (view_name, sql) pairs, in alphabetical order."""
    files = sorted(QUERIES_DIR.glob("*.sql"))
    if not files:
        print(f"No .sql files found in {QUERIES_DIR}", file=sys.stderr)
        sys.exit(1)

    result = []
    for f in files:
        name = view_name_from_file(f)
        if only and name not in only:
            continue
        body = f.read_text()
        result.append((name, build_view_sql(name, body)))
    return result


def apply_to_db(views: list[tuple[str, str]], db_url: str) -> None:
    try:
        import psycopg2
    except ImportError:
        print(
            "psycopg2 not installed. Run: pip install psycopg2-binary",
            file=sys.stderr,
        )
        sys.exit(1)

    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        for name, sql in views:
            print(f"  Creating view: {SCHEMA}.{name} ...", end=" ")
            cur.execute(sql)
            print("OK")
        # Also refresh grants so the readonly role can see new views
        cur.execute(
            f"GRANT SELECT ON ALL TABLES IN SCHEMA {SCHEMA} TO analytics_readonly;"
        )
        conn.commit()
        print(f"\n✓ {len(views)} view(s) created/updated successfully.")
    except Exception as e:
        conn.rollback()
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--setup", action="store_true", help="Print one-time schema/role/grant SQL"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print SQL without executing"
    )
    parser.add_argument(
        "--db-url", help="Postgres connection string (overrides DATABASE_URL)"
    )
    parser.add_argument(
        "--only", help="Comma-separated list of view names to update (default: all)"
    )
    args = parser.parse_args()

    if args.setup:
        print(SETUP_SQL)
        return

    only = [v.strip() for v in args.only.split(",")] if args.only else None
    views = generate_all(only=only)

    if not views:
        print("No matching views found.")
        sys.exit(0)

    if args.dry_run:
        print(f"-- Generated by generate_views.py ({len(views)} views)\n")
        for name, sql in views:
            print(f"-- ── {name} ──────────────────────────────")
            print(sql)
        return

    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print(
            "No database URL provided.\n"
            "Use --db-url or set DATABASE_URL environment variable.\n"
            "Use --dry-run to just print the SQL.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Applying {len(views)} view(s) to database...")
    apply_to_db(views, db_url)


if __name__ == "__main__":
    main()
