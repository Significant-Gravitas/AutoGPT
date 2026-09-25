"""Apply the real catalogue migration to an empty, explicitly local test database."""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import psycopg2

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parents[1]
MIGRATION = Path(
    "autogpt_platform/backend/migrations/20260925190000_catalogue_release_publisher/migration.sql"
)
SCHEMA = "autogpt_platform/backend/schema.prisma"


def main() -> None:
    url = os.environ["CATALOGUE_TEST_DATABASE_URL"]
    parsed = urlparse(url)
    if parsed.hostname not in {"localhost", "127.0.0.1"} or not re.fullmatch(
        r"/catalogue_[a-z0-9_]+", parsed.path
    ):
        raise ValueError("Only a dedicated local catalogue_* test database is allowed")
    if parse_qs(parsed.query).get("schema") != ["platform"]:
        raise ValueError("The disposable database must use schema=platform")
    connection = psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        user=parsed.username,
        password=parsed.password,
        dbname=parsed.path[1:],
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM pg_tables WHERE schemaname NOT IN ('pg_catalog','information_schema')"
            )
            table_count = cursor.fetchone()
            if table_count is None or table_count[0] != 0:
                raise ValueError(
                    "Test database is not empty; create a new disposable database"
                )
        environment = {**os.environ, "DATABASE_URL": url, "DIRECT_URL": url}
        with tempfile.TemporaryDirectory(prefix="catalogue-schema-") as directory:
            baseline = Path(directory) / "schema.prisma"
            baseline.write_bytes(_base_schema())
            _prisma(
                environment, "db", "push", "--skip-generate", "--schema", str(baseline)
            )
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SET search_path TO platform, public")
                cursor.execute((REPO / MIGRATION).read_text(encoding="utf-8"))
        _prisma(environment, "db", "push", "--skip-generate")
    finally:
        connection.close()
    print("Exact catalogue migration applied to an empty local test database")


def _base_schema() -> bytes:
    introduced = (
        subprocess.check_output(
            [
                "git",
                "log",
                "--format=%H",
                "--diff-filter=A",
                "-1",
                "--",
                str(MIGRATION),
            ],
            cwd=REPO,
        )
        .decode()
        .strip()
    )
    ref = f"{introduced}^" if introduced else "HEAD"
    return subprocess.check_output(["git", "show", f"{ref}:{SCHEMA}"], cwd=REPO)


def _prisma(environment: dict[str, str], *arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "prisma", *arguments],
        cwd=BACKEND,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
