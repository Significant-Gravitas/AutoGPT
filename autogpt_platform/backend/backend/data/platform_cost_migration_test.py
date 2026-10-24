"""The PlatformCostLog ``expertId`` migrations keep the shape that lets the
index build run without blocking cost-log writes.

Postgres refuses ``CREATE INDEX CONCURRENTLY`` inside a transaction block,
and Prisma puts a migration with more than one statement in an implicit
transaction, so the index is the only statement of its own migration. A
plain ``CREATE INDEX`` would hold a SHARE lock and block every cost-log
write (every chat turn and graph run writes one) for the whole scan.
"""

import re
from pathlib import Path

_MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


def _statements(migration: str) -> list[str]:
    """The migration's executable statements, comments stripped and
    whitespace collapsed."""
    sql = _COMMENTS.sub("", (_MIGRATIONS / migration / "migration.sql").read_text())
    return [" ".join(part.split()) for part in sql.split(";") if part.strip()]


def test_the_index_is_built_concurrently_as_its_migrations_only_statement() -> None:
    assert _statements("20260926160001_platform_cost_log_expert_id_index") == [
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS "PlatformCostLog_expertId_createdAt_idx" '
        'ON "PlatformCostLog"("expertId", "createdAt")'
    ]


def test_the_column_migration_builds_no_index() -> None:
    assert _statements("20260926160000_platform_cost_log_expert_id") == [
        'ALTER TABLE "PlatformCostLog" ADD COLUMN IF NOT EXISTS "expertId" TEXT'
    ]
