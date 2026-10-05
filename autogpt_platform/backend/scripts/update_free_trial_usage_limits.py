"""Apply the $20 lifetime budget to stored free-trial offers.

Before deploying the new offer policy, pause new trial enrollments with the
offer flag's ``max_active_trials=0`` and retain its previous value. Deploy
every backend worker and drain old workers, run this backfill, then restore
the previous enrollment cap. Alternatively, first deploy a validator-only
compatibility release to all workers. Old validators reject a daily limit
above the weekly limit, including offers created during a rolling deploy.
This is intentionally a post-deploy backfill, not a Prisma migration.
Usage and existing trial dates are preserved.

Set DATABASE_URL for the target environment, then run from the backend::

    poetry run python -m scripts.update_free_trial_usage_limits
    poetry run python -m scripts.update_free_trial_usage_limits --apply

The default dry run only counts offers needing an update. Re-running
``--apply`` is safe; only the three stored usage limits are changed.
"""

import argparse
import asyncio
import os
from pathlib import Path

from pydantic import BaseModel

from backend.data.db import (
    connect,
    disconnect,
    execute_raw_with_schema,
    query_raw_with_schema,
)

SQL_PATH = Path(__file__).with_name("sql") / "update_free_trial_usage_limits.sql"


class OfferCount(BaseModel):
    count: int


async def main(apply: bool = False) -> int:
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL must be set")
    update_sql = SQL_PATH.read_text(encoding="utf-8")
    count_sql = _count_query(update_sql)
    try:
        await connect()
        if apply:
            updated = await execute_raw_with_schema(update_sql)
            print(f"Updated {updated} free-trial offers.")
        else:
            rows = await query_raw_with_schema(count_sql, model=OfferCount)
            print(f"DRY RUN: {rows[0].count} free-trial offers need an update.")
            print("No changes made. Re-run with --apply after all workers are updated.")
    finally:
        await disconnect()
    return 0


def _count_query(update_sql: str) -> str:
    _, separator, predicate = update_sql.partition("\nWHERE ")
    if not separator:
        raise ValueError("The trial backfill SQL must contain a WHERE clause")
    return (
        'SELECT COUNT(*)::int AS count FROM {schema_prefix}"SubscriptionTrial"'
        f"\nWHERE {predicate}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update matching offers; default is dry run.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(asyncio.run(main(apply=args.apply)))
