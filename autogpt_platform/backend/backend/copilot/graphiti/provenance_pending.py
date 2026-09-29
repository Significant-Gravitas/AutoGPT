"""The graphs holding a dream citation marker whose record failed, and the
sweep the dream reaper runs over them.

When a dream write's derivation record fails after its graph write, or the
graph write itself raises after its marker was written, the ingestion worker
(``marked_write.py``) notes the graph here: a Redis set of graph names,
``PENDING_KEY``. Each run of the dream reaper (``dream/reaper.py``) sweeps
up to ``SWEEP_MAX_GRAPHS`` of them: it takes the graph's write lock,
completes its markers (``recall_reconcile.reconcile``) and, once none is
left, removes the graph from the set while it still holds the lock, so a
failure noted meanwhile (also under that lock) is never lost. A graph it
cannot lock, or whose reconcile fails, stays for the next run.

This is the background path; the guarantee does not rest on it. A marker no
failure noted (a worker that died between its marker and its record) and one
whose note was lost (Redis unreachable) are completed by the next forget in
their graph, which reconciles before it cascades (``recall_forget.py``), or
by the derivation backfill.
"""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from backend.data import redis_client

from .falkordb_driver import open_graph_driver
from .recall_reconcile import reconcile
from .scope_lock import LockState, graph_write_lock

logger = logging.getLogger(__name__)

PENDING_KEY = "graphiti:provenance_pending"
SWEEP_MAX_GRAPHS = 20
SWEEP_LOCK_WAIT_SECONDS = 5
_REDIS_TIMEOUT_SECONDS = 2.0


class Swept(BaseModel):
    """One sweep: the graphs it reconciled, the records it completed, and
    the graphs left for the next run, ``busy`` or ``failed``."""

    graphs: int = 0
    completed: int = 0
    busy: int = 0
    failed: int = 0


async def note_pending(group_id: str) -> None:
    """Note graph ``group_id`` as holding a marker to reconcile; never
    raises (a lost note leaves the marker to the next forget)."""
    try:
        redis = await _redis()
        await asyncio.wait_for(
            redis.sadd(PENDING_KEY, group_id), _REDIS_TIMEOUT_SECONDS
        )
    except Exception:
        logger.warning(
            f"Could not note graph {group_id[:20]} as holding a pending dream "
            "record; the next forget there completes it",
            exc_info=True,
        )


async def sweep_pending(*, max_graphs: int = SWEEP_MAX_GRAPHS) -> Swept:
    """Reconcile up to ``max_graphs`` of the noted graphs; never raises."""
    swept = Swept()
    try:
        redis = await _redis()
        graphs = await asyncio.wait_for(
            redis.srandmember(PENDING_KEY, max_graphs), _REDIS_TIMEOUT_SECONDS
        )
    except Exception:
        logger.warning("Could not read the graphs with pending dream records")
        return swept
    for group_id in graphs or []:
        await _sweep_graph(redis, str(group_id), swept)
    return swept


async def _sweep_graph(redis: Any, group_id: str, swept: Swept) -> None:
    """Reconcile one graph under its write lock; forget the note once no
    marker is left, still holding the lock."""
    try:
        async with graph_write_lock(
            group_id, wait_seconds=SWEEP_LOCK_WAIT_SECONDS
        ) as lock:
            if lock is not LockState.HELD:
                swept.busy += 1
                return
            driver = open_graph_driver(group_id)
            try:
                done = await reconcile(driver, group_id)
            finally:
                await driver.close()
            swept.graphs += 1
            swept.completed += done.completed
            if not (done.left or done.waiting):
                await asyncio.wait_for(
                    redis.srem(PENDING_KEY, group_id), _REDIS_TIMEOUT_SECONDS
                )
    except Exception:
        logger.warning(
            f"Could not reconcile graph {group_id[:20]}; kept for the next run",
            exc_info=True,
        )
        swept.failed += 1


async def _redis() -> Any:
    return await asyncio.wait_for(
        redis_client.get_redis_async(), _REDIS_TIMEOUT_SECONDS
    )
