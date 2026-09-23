"""Load the Skills Hub catalog and the expert roster as a deploy step.

Run with: poetry run seed-experts

(or ``python3 -m backend.api.features.experts.deploy_seed`` in the image)

Meant for the deploy's post-upgrade seed Job (``autogpt-server`` Helm chart,
``templates/job-seed.yaml`` in the infrastructure repo), which runs it in the
freshly deployed image after the platform repo has applied the migrations, so
a roster or catalog change reaches an environment with the code that ships it
rather than waiting for someone to seed by hand. It is just as safe to run by
hand against an environment.

Three guarantees make it safe to run on every deploy:

* It refuses to start while any migration in ``migrations/`` is unapplied or
  failed. The seeds write columns the latest migrations add, so a schema that
  is behind would fail part-way; checking first means nothing is written.
* One run at a time per database, via a transaction-scoped advisory lock. Two
  overlapping deploys would otherwise both miss a new template by name and
  both create it. The second run waits, then converges on the same rows.
* Both seeds are upserts keyed on stable identifiers (skill slug, template
  name), and neither overwrites what a hire's owner changed. Running it twice
  leaves the second run with nothing to change.

Order matters. The workflows the roster preloads have to be published on the
environment's marketplace already, and nothing here can publish them, so they
are checked before anything is written: a missing one fails the run, names
the slugs, and leaves the database as it was. The skills catalog goes next,
because the roster refuses a template whose bundled skill has no Skills Hub
listing. A failure exits non-zero, which fails the seed Job; re-running after
the fix converges, because every write is an upsert.
"""

import asyncio
import logging
from datetime import timedelta
from pathlib import Path

from backend.api.features.experts import seed
from backend.api.features.store import skill_seed
from backend.data import db as database

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parents[4] / "migrations"
# Arbitrary but fixed: every caller of this module must contend on one key.
SEED_LOCK_KEY = 0x5EED_E4E7
# Held for the whole run, so it has to outlast a full catalog download and
# roster seed against a remote database, plus waiting out an overlapping run.
SEED_LOCK_TIMEOUT = timedelta(minutes=30)


class SchemaBehindError(RuntimeError):
    pass


def migrations_on_disk(directory: Path | None = None) -> set[str]:
    return {
        entry.name
        for entry in (directory or MIGRATIONS_DIR).iterdir()
        if entry.is_dir() and (entry / "migration.sql").is_file()
    }


async def assert_schema_current(directory: Path | None = None) -> None:
    """Raise unless every migration this build ships is applied cleanly."""
    rows = await database.query_raw_with_schema(
        'SELECT migration_name, finished_at, rolled_back_at FROM {schema_prefix}"_prisma_migrations"'
    )
    applied = {
        row["migration_name"]
        for row in rows
        if row["finished_at"] is not None and row["rolled_back_at"] is None
    }
    failed = sorted(
        row["migration_name"]
        for row in rows
        if row["finished_at"] is None and row["rolled_back_at"] is None
    )
    pending = sorted(migrations_on_disk(directory) - applied)
    if failed or pending:
        raise SchemaBehindError(
            "Refusing to seed experts: the database schema is not current "
            f"(failed: {', '.join(failed) or 'none'}; "
            f"pending: {', '.join(pending) or 'none'}). "
            "Run `prisma migrate deploy` first."
        )


async def seed_experts() -> list[str]:
    """Seed the skills catalog, then the roster. Returns the template ids."""
    await assert_schema_current()
    async with database.transaction(timeout=SEED_LOCK_TIMEOUT) as lock:
        await lock.execute_raw(
            "SELECT pg_advisory_xact_lock($1::bigint)", SEED_LOCK_KEY
        )
        await seed._resolve_roster_preloads()
        skills = await skill_seed.seed_catalog_skills()
        logger.info(f"Seeded {len(skills)} Skills Hub listings")
        template_ids = await seed.seed_roster()
        logger.info(f"Seeded {len(template_ids)} expert templates")
    return template_ids


async def main() -> None:
    await database.connect()
    try:
        await seed_experts()
    finally:
        await database.disconnect()


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())


if __name__ == "__main__":
    run()
