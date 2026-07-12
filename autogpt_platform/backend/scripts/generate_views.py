#!/usr/bin/env python3
"""
AutoGPT Analytics — View Generator
====================================
Reads every .sql file in analytics/queries/ and registers it as a
CREATE OR REPLACE VIEW in the analytics schema.

Quick start (from autogpt_platform/backend/):

Step 1 — one-time setup (creates schema, role, grants):

  poetry run analytics-setup

Step 2 — create / refresh all 14 analytics views:

  poetry run analytics-views

Both commands auto-detect credentials from .env (DB_* vars).
Use --db-url to override.

Step 3 (optional) — enable login and set a password for the read-only
role so external tools (Supabase MCP, PostHog Data Warehouse) can connect.
The role is created as NOLOGIN, so you must grant LOGIN at the same time.
Run in Supabase SQL Editor:

  ALTER ROLE analytics_readonly WITH LOGIN PASSWORD 'your-password';

Usage
-----
  poetry run analytics-setup              # apply setup to DB
  poetry run analytics-setup --dry-run   # print setup SQL only
  poetry run analytics-views             # apply all views to DB
  poetry run analytics-views --dry-run   # print all view SQL only
  poetry run analytics-views --only graph_execution,retention_login_weekly

Environment variables
---------------------
  DATABASE_URL   Postgres connection string (checked before .env)

Notes
-----
- .env DB_* vars are read automatically as a fallback.
- Safe to re-run: uses CREATE OR REPLACE VIEW.
- Looker, PostHog Data Warehouse, and Supabase MCP all read from the
  same analytics.* views — no raw tables exposed.
"""

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import quote

BACKEND_DIR = Path(__file__).parent.parent
QUERIES_DIR = BACKEND_DIR.parent / "analytics" / "queries"
ENV_FILE = BACKEND_DIR / ".env"
SCHEMA = "analytics"

SETUP_SQL = """\
-- =============================================================
-- AutoGPT Analytics Schema Setup
-- Run ONCE as the postgres superuser (e.g. via Supabase SQL Editor).
-- After this, run: poetry run analytics-views
-- =============================================================

-- 1. Create the analytics schema
CREATE SCHEMA IF NOT EXISTS analytics;

-- 2. Create the read-only role (skip if already exists)
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'analytics_readonly') THEN
    CREATE ROLE analytics_readonly NOLOGIN;
  END IF;
END
$$;

-- 3. Analytics schema grants only.
--    Views use security_invoker = false so they execute as their
--    owner (postgres). analytics_readonly never needs direct access
--    to the platform or auth schemas.
GRANT USAGE ON SCHEMA analytics TO analytics_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO analytics_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics
  GRANT SELECT ON TABLES TO analytics_readonly;
"""


def load_db_url_from_env() -> str | None:
    """Read DB_* vars from .env and build a psycopg2 connection string."""
    if not ENV_FILE.exists():
        return None
    env: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    host = env.get("DB_HOST", "localhost")
    port = env.get("DB_PORT", "5432")
    user = env.get("DB_USER", "postgres")
    password = env.get("DB_PASS", "")
    dbname = env.get("DB_NAME", "postgres")
    if not password:
        return None
    return (
        "postgresql://"
        f"{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(dbname, safe='')}"
    )


def get_db_url(args: argparse.Namespace) -> str | None:
    return args.db_url or os.environ.get("DATABASE_URL") or load_db_url_from_env()


def connect(db_url: str):
    try:
        import psycopg2
    except ImportError:
        print("psycopg2 not found. Run: poetry install", file=sys.stderr)
        sys.exit(1)
    return psycopg2.connect(db_url)


def run_sql(db_url: str, statements: list[tuple[str, str]]) -> None:
    """Execute a list of (label, sql) pairs in a single transaction."""
    conn = connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()
    try:
        for label, sql in statements:
            print(f"  {label} ...", end=" ")
            cur.execute(sql)
            print("OK")
        conn.commit()
        print(f"\n✓ {len(statements)} statement(s) applied.")
    except Exception as e:
        conn.rollback()
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


def build_view_sql(name: str, query_body: str) -> str:
    body = query_body.strip().rstrip(";")
    # security_invoker = false → view runs as its owner (postgres), not the
    # caller, so analytics_readonly only needs analytics schema access.
    return f"CREATE OR REPLACE VIEW {SCHEMA}.{name} WITH (security_invoker = false) AS\n{body};\n"


def load_views(only: list[str] | None = None) -> list[tuple[str, str]]:
    """Return [(label, sql)] for all views, in alphabetical order."""
    files = sorted(QUERIES_DIR.glob("*.sql"))
    if not files:
        print(f"No .sql files found in {QUERIES_DIR}", file=sys.stderr)
        sys.exit(1)
    known = {f.stem for f in files}
    if only:
        unknown = [n for n in only if n not in known]
        if unknown:
            print(
                f"Unknown view name(s): {', '.join(unknown)}\n"
                f"Available: {', '.join(sorted(known))}",
                file=sys.stderr,
            )
            sys.exit(1)
    result = []
    for f in files:
        name = f.stem
        if only and name not in only:
            continue
        result.append((f"view analytics.{name}", build_view_sql(name, f.read_text())))
    return result


def no_db_url_error() -> None:
    print(
        "No database URL found.\n"
        "Tried: --db-url, DATABASE_URL env var, and .env (DB_* vars).\n"
        "Use --dry-run to just print the SQL.",
        file=sys.stderr,
    )
    sys.exit(1)


def cmd_setup(args: argparse.Namespace) -> None:
    if args.dry_run:
        print(SETUP_SQL)
        return
    db_url = get_db_url(args)
    if not db_url:
        no_db_url_error()
    assert db_url
    print("Applying analytics setup...")
    run_sql(db_url, [("schema / role / grants", SETUP_SQL)])


def cmd_views(args: argparse.Namespace) -> None:
    only = [v.strip() for v in args.only.split(",")] if args.only else None
    views = load_views(only=only)
    if not views:
        print("No matching views found.")
        sys.exit(0)

    if args.dry_run:
        print(f"-- {len(views)} views\n")
        for label, sql in views:
            print(f"-- {label}")
            print(sql)
        return

    db_url = get_db_url(args)
    if not db_url:
        no_db_url_error()
    assert db_url
    print(f"Applying {len(views)} view(s)...")
    # Append grant refresh so the readonly role sees any new views
    grant = f"GRANT SELECT ON ALL TABLES IN SCHEMA {SCHEMA} TO analytics_readonly;"
    run_sql(db_url, views + [("grant analytics_readonly", grant)])


def main_setup() -> None:
    parser = argparse.ArgumentParser(description="Apply analytics schema setup to DB")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print SQL, don't execute"
    )
    parser.add_argument("--db-url", help="Postgres connection string")
    cmd_setup(parser.parse_args())


def main_views() -> None:
    parser = argparse.ArgumentParser(description="Apply analytics views to DB")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print SQL, don't execute"
    )
    parser.add_argument("--db-url", help="Postgres connection string")
    parser.add_argument("--only", help="Comma-separated view names to update")
    cmd_views(parser.parse_args())


if __name__ == "__main__":
    # Default: apply views (backwards-compatible with direct python invocation)
    main_views()
