"""A dream write between its citation marker and its derivation record: the
ingestion worker's side of ``recall_derivation.py``.

``ingest._write_locked`` holds the graph's write lock for the whole write
and calls these in turn. ``marked`` writes the write's citation marker
before the graph write; when it cannot, the write is not made and counts as
failed (it fails closed). ``added`` makes the graph write; one that raised
after its marker may have landed in part, so its graph is noted for the
reaper (``provenance_pending.py``). ``recorded`` records what the write was
derived from; a record that fails leaves the marker, notes the graph and
counts as ``provenance_pending``. The counts land on the pass's
``ingest.IngestionCompletion``.
"""

import logging
from collections.abc import Awaitable
from typing import Any, Protocol

from graphiti_core import Graphiti
from graphiti_core.graphiti import AddEpisodeResults

from .provenance_pending import note_pending
from .recall_citations import Citations
from .recall_derivation import mark, record

logger = logging.getLogger(__name__)


class Outcomes(Protocol):
    """The counters a dream write's outcome lands on
    (``ingest.IngestionCompletion``)."""

    failed: int
    provenance_pending: int


async def marked(
    client: Graphiti,
    group_id: str,
    payload: dict[str, Any],
    citations: Citations | None,
    outcomes: Outcomes | None,
) -> str | None:
    """The citation marker of a dream write, written before the write; None
    for any other write, and for a dream write whose marker could not be
    written, which is then not made and counts as failed."""
    if citations is None:
        return None
    try:
        return await mark(client.driver, group_id, str(payload["name"]), citations)
    except Exception:
        logger.warning(
            f"Dropped dream write {payload.get('name')!r}: its citation marker "
            f"could not be written in graph {group_id[:20]}",
            exc_info=True,
        )
        if outcomes is not None:
            outcomes.failed += 1
        return None


async def added(
    write: Awaitable[AddEpisodeResults], group_id: str, marker: str | None
) -> AddEpisodeResults:
    """The graph ``write``, awaited; one that raised after its ``marker`` may
    have landed in part, so its graph is noted for the reaper to reconcile."""
    try:
        return await write
    except Exception:
        if marker is not None:
            await note_pending(group_id)
        raise


async def recorded(
    client: Graphiti,
    group_id: str,
    marker: str,
    result: AddEpisodeResults,
    citations: Citations,
    outcomes: Outcomes | None,
) -> None:
    """Record what the dream write just made was derived from; a record that
    failed leaves its marker, is noted for the reaper and counts as
    ``provenance_pending``."""
    episode = result.episode.uuid
    touched = [edge.uuid for edge in result.edges if episode in edge.episodes]
    if await record(client.driver, group_id, marker, episode, touched, citations):
        return
    await note_pending(group_id)
    if outcomes is not None:
        outcomes.provenance_pending += 1
