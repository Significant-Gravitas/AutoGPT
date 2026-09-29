"""A dream write between its citation marker and its derivation record: the
ingestion worker's side of ``recall_derivation.py``.

``ingest._write_locked`` holds the graph's write lock for the whole write
and calls these in turn. ``marked`` draws the uuid the write's episode will
have and writes the write's citation marker under it before anything else;
when the marker cannot be written the write is not made and counts as
failed (it fails closed). A write the check against forgets then drops has
its marker taken back (``withdrawn``). ``placed`` puts the episode in the
graph under that uuid, ``write_pending`` (hidden from recall), and
graphiti's ``add_episode`` is handed the uuid: graphiti-core 0.30.2 loads
the episode it names (``EpisodicNode.get_by_uuid``) instead of creating
one, extracts from it, and saves it again with its facts (a ``MERGE`` on
the uuid that replaces every property), so ``write_pending`` goes when
graphiti saves the episode. ``added`` runs the write; one that raised marks
its marker ``aborted`` and notes the graph for the reaper
(``provenance_pending.py``), as the episode may have landed in part.
``recorded`` records what the write was derived from and settles it
(``recall_derivation.record``); a record that fails leaves the marker,
notes the graph and counts as ``provenance_pending``. The counts land on
the pass's ``ingest.IngestionCompletion``.
"""

import logging
import uuid as uuidlib
from collections.abc import Awaitable
from datetime import datetime, timezone
from typing import Any, Protocol

from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.graphiti import AddEpisodeResults
from pydantic import BaseModel

from .provenance_pending import note_pending
from .recall_citations import Citations
from .recall_derivation import abort, mark, record, withdraw

logger = logging.getLogger(__name__)


class Outcomes(Protocol):
    """The counters a dream write's outcome lands on
    (``ingest.IngestionCompletion``)."""

    failed: int
    provenance_pending: int


class Marked(BaseModel):
    """A dream write's marker and the uuid its episode is written under."""

    marker: str
    episode: str


async def marked(
    client: Graphiti,
    group_id: str,
    payload: dict[str, Any],
    citations: Citations | None,
    outcomes: Outcomes | None,
) -> Marked | None:
    """The citation marker of a dream write, written before the write under
    the episode uuid drawn for it; None for any other write, and for a dream
    write whose marker could not be written, which is then not made and
    counts as failed."""
    if citations is None:
        return None
    episode = str(uuidlib.uuid4())
    try:
        marker = await mark(
            client.driver, group_id, episode, str(payload["name"]), citations
        )
    except Exception:
        logger.warning(
            f"Dropped dream write {payload.get('name')!r}: its citation marker "
            f"could not be written in graph {group_id[:20]}",
            exc_info=True,
        )
        if outcomes is not None:
            outcomes.failed += 1
        return None
    return Marked(marker=marker, episode=episode)


async def withdrawn(driver: GraphDriver, marked: Marked | None) -> None:
    """Take back the marker of a dream write dropped before it was made."""
    if marked is not None:
        await withdraw(driver, marked.marker)


async def placed(
    driver: GraphDriver, group_id: str, payload: dict[str, Any], episode: str
) -> None:
    """Put the write's episode in the graph under ``episode``, as graphiti
    saves one and ``write_pending``, for graphiti's ``add_episode`` to load
    and save over."""
    await driver.execute_query(
        PLACE_EPISODE_QUERY,
        uuid=episode,
        name=str(payload["name"]),
        group_id=group_id,
        source_description=payload["source_description"],
        source=payload["source"].value,
        content=payload["episode_body"],
        created_at=datetime.now(timezone.utc),
        valid_at=payload["reference_time"],
    )


async def added(
    write: Awaitable[AddEpisodeResults],
    driver: GraphDriver,
    group_id: str,
    marked: Marked | None,
) -> AddEpisodeResults:
    """The graph ``write``, awaited; a marked one that raised may have landed
    in part, so its marker is marked aborted and its graph noted for the
    reaper to reconcile."""
    try:
        return await write
    except Exception:
        if marked is not None:
            await abort(driver, marked.marker)
            await note_pending(group_id)
        raise


async def recorded(
    client: Graphiti,
    group_id: str,
    marked: Marked,
    result: AddEpisodeResults,
    citations: Citations,
    outcomes: Outcomes | None,
) -> None:
    """Record what the dream write just made was derived from and settle it;
    a record that failed leaves its marker, is noted for the reaper and
    counts as ``provenance_pending``."""
    episode = result.episode.uuid
    touched = [edge.uuid for edge in result.edges if episode in edge.episodes]
    if await record(
        client.driver, group_id, marked.marker, episode, touched, citations
    ):
        return
    await note_pending(group_id)
    if outcomes is not None:
        outcomes.provenance_pending += 1


# The properties graphiti saves an episode with (graphiti-core 0.30.2), and
# ``write_pending``, which graphiti's own save of the episode replaces.
PLACE_EPISODE_QUERY = """
CREATE (:Episodic {
    uuid: $uuid,
    name: $name,
    group_id: $group_id,
    source_description: $source_description,
    source: $source,
    content: $content,
    entity_edges: [],
    created_at: $created_at,
    valid_at: $valid_at,
    write_pending: true
})
"""
