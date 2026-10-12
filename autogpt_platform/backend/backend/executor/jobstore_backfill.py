"""Rewrite enum members pickled into persisted APScheduler job arguments.

Job kwargs written before #13190 used ``model_dump()`` rather than
``model_dump(mode="json")``, so real Enum members were pickled into
``apscheduler_jobs``. ``ProviderName`` is one of them, and an Enum pickles by
value: unpickling calls ``ProviderName("github")``, which raises the moment a
member is renamed or dropped — and APScheduler's answer to a job it cannot
restore is to destroy it (SENTRY-1392).

This converts those members to their plain values, which the dispatch
functions' pydantic models coerce back on the way in. It must run while every
member still resolves, i.e. BEFORE any deploy that removes one.

    poetry run python -m backend.executor.jobstore_backfill            # dry run
    poetry run python -m backend.executor.jobstore_backfill --apply
"""

import argparse
import logging
import os
import pickle
import sys
from enum import Enum
from typing import Any

from sqlalchemy import MetaData, Table, create_engine, select
from sqlalchemy.exc import NoSuchTableError

logger = logging.getLogger(__name__)

TABLES = ("apscheduler_jobs", "apscheduler_jobs_batched_notifications")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the changes; without it the run only reports",
    )
    args = parser.parse_args()
    # Imported here: scheduler pulls in the whole backend, and this
    # module is imported by tests that need none of it.
    from backend.executor.scheduler import _extract_schema_from_url

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    database_url = os.getenv("DIRECT_URL")
    if not database_url:
        logger.error("DIRECT_URL is not set")
        return 2

    schema, url = _extract_schema_from_url(database_url)
    engine = create_engine(url)
    metadata = MetaData(schema=schema)

    total_changed = total_unreadable = total_skipped = 0
    for tablename in TABLES:
        changed, unreadable, skipped = _normalize_table(
            engine, metadata, tablename, args.apply
        )
        total_changed += changed
        total_unreadable += unreadable
        total_skipped += skipped

    verb = "rewrote" if args.apply else "would rewrite"
    logger.info(f"\n{verb} {total_changed} row(s); {total_unreadable} unreadable")
    if not args.apply and total_changed:
        logger.info("re-run with --apply to write")
    if total_skipped:
        # The run is incomplete; saying so here is the point, since this is the
        # line an operator reads to decide the backfill is done.
        logger.warning(
            f"{total_skipped} row(s) changed under the scan and were NOT rewritten; "
            "re-run to pick them up"
        )
        return 1
    return 0


def _normalize_table(
    engine, metadata, tablename: str, apply: bool
) -> tuple[int, int, int]:
    try:
        table = Table(tablename, metadata, autoload_with=engine)
    except NoSuchTableError:
        logger.info(f"{tablename}: skipped (no such table)")
        return 0, 0, 0

    changed = unreadable = skipped = 0
    with engine.begin() as connection:
        rows = list(
            connection.execute(
                select(table.c.id, table.c.job_state, table.c.next_run_time)
            )
        )
        for row in rows:
            try:
                state = pickle.loads(row.job_state)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                # Already unrestorable; parking/repair is a separate concern.
                logger.warning(f"{tablename}: {row.id} is unreadable ({e})")
                unreadable += 1
                continue

            args, kwargs = state.get("args", ()), state.get("kwargs", {})
            # Not an equality check: a str-based Enum compares equal to its own
            # value, so the stripped copy would look unchanged and be skipped.
            if not (_has_enum(args) or _has_enum(kwargs)):
                continue

            new_state = dict(state)
            new_state["args"] = _strip_enums(args)
            new_state["kwargs"] = _strip_enums(kwargs)
            if row.next_run_time is None:
                # The row is paused, and parking is what pauses an unrestorable
                # one -- but parking could only clear the COLUMN, so this copy
                # still holds the old time. Leaving them disagreed strands the
                # row: nothing reports it, nothing runs it, resume refuses it.
                new_state["next_run_time"] = None

            logger.info(f"{tablename}: {row.id} carries pickled enum(s)")
            if not apply:
                changed += 1
                continue

            result = connection.execute(
                table.update()
                .where(
                    table.c.id == row.id,
                    table.c.job_state == row.job_state,
                )
                .values(job_state=pickle.dumps(new_state, pickle.HIGHEST_PROTOCOL))
            )
            if result.rowcount == 1:
                changed += 1
            else:
                skipped += 1
                logger.warning(
                    f"{tablename}: {row.id} changed under the scan; left alone, "
                    "re-run to pick it up"
                )

    suffix = f", {skipped} skipped (changed under the scan)" if skipped else ""
    logger.info(
        f"{tablename}: {len(rows)} row(s) scanned, {changed} to rewrite{suffix}"
    )
    return changed, unreadable, skipped


def _has_enum(value: Any) -> bool:
    """Whether an Enum member is hiding anywhere in *value*."""
    if isinstance(value, Enum):
        return True
    if isinstance(value, dict):
        return any(_has_enum(k) or _has_enum(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_enum(v) for v in value)
    return False


def _strip_enums(value: Any) -> Any:
    """Replace Enum members with their values, recursively."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {_strip_enums(k): _strip_enums(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_enums(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_strip_enums(v) for v in value)
    return value


if __name__ == "__main__":
    sys.exit(main())
