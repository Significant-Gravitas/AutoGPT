"""How long the DreamPass table keeps a pass, and the job that enforces it.

A row sheds most of its weight as it closes: every closing transition drops
its input bundle with its lease (``pass_record.py``), keeping the phase
outputs, the operations and the usage. The bundle is the bulk of an open
batch row: Codex measured one gathered from the dream fixtures (50 episodes,
500 facts, 10 session bodies) at 167 KB of JSON, which Postgres compressed to
7.2 KB, about 8.7 KB of table per row all told; real text compresses less,
and a bundle has no size cap, so an uncompressed one can run to hundreds of
KB. A closed row keeps the smaller JSON only, a few KB.

Closed rows are kept ``Config.dream_pass_retention_days`` (the
``DREAM_PASS_RETENTION_DAYS`` setting, 90 by default) from their creation;
a weekly scheduler job deletes the older ones and logs how many went. An open
row is never deleted, however old (closing it is the reaper's job), nor a
closed one whose cleanup the reaper has yet to finish.

Bounded: ``RETENTION_BATCH_SIZE`` rows per statement, each statement
``RETENTION_BATCH_TIMEOUT_SECONDS``, at most ``RETENTION_MAX_BATCHES``
statements and ``RETENTION_BUDGET_SECONDS`` a run; the next run goes on from
there. Not bounded: how much of the table a statement reads to find its
batch. There is no index on ``createdAt`` alone, so each statement scans
the table until it has its batch: cheap while the table is small, and an
index to add should it grow.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from .store import delete_old_passes

logger = logging.getLogger(__name__)

RETENTION_BATCH_SIZE = 1000
RETENTION_MAX_BATCHES = 100
# One batch's delete, rows and their compressed JSON included.
RETENTION_BATCH_TIMEOUT_SECONDS = 60.0
# One run, however many batches it gets through.
RETENTION_BUDGET_SECONDS = 900.0


async def delete_expired_records(
    retention_days: int, *, now: datetime | None = None
) -> int:
    """Delete the closed passes created more than *retention_days* ago, a
    batch at a time, and say how many went. Never raises: a batch that fails
    or a run out of budget stops it, logged, and the next run goes on."""
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=retention_days)
    deleted = 0
    try:
        async with asyncio.timeout(RETENTION_BUDGET_SECONDS):
            for _ in range(RETENTION_MAX_BATCHES):
                batch = await delete_old_passes(
                    cutoff,
                    limit=RETENTION_BATCH_SIZE,
                    timeout=RETENTION_BATCH_TIMEOUT_SECONDS,
                )
                deleted += batch
                if batch < RETENTION_BATCH_SIZE:
                    break
    except Exception:
        logger.warning(
            f"Dream pass retention stopped after {deleted} deleted", exc_info=True
        )
    logger.info(
        f"Dream pass retention: deleted {deleted} closed pass(es) created "
        f"before {cutoff.isoformat()} ({retention_days}-day retention)"
    )
    return deleted
