"""Seeding and reads for the live dream-marker tests
(``recall_marker_integration_test.py``,
``recall_marker_crash_integration_test.py``,
``recall_marker_race_integration_test.py``,
``recall_ancestry_integration_test.py``): a user's fact and the episode
that stated it, a write landing as graphiti saves one, the markers a graph
holds, and a forget run to the end while graphiti saves a dream write.
Not collected by pytest.
"""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

from graphiti_core.driver.falkordb_driver import FalkorDriverSession
from graphiti_core.nodes import EpisodeType

from .falkordb_driver import AutoGPTFalkorDriver
from .memory_model import ForgetResult
from .recall_derivation import MARKER_LABEL
from .recall_forget import retract
from .recall_integration_fixtures import rows
from .scope import MemoryScope

# Long past ``recall_reconcile.MARKER_EXPIRY_SECONDS``.
LONG_AGO = "2000-01-01T00:00:00+00:00"


async def seed_root(driver: AutoGPTFalkorDriver, group_id: str, root: str) -> None:
    """A user's fact ``root`` and the episode that stated it."""
    await driver.execute_query(
        """
        CREATE (a:Entity {uuid: $root + '-a', name: 'Root A', group_id: $gid}),
               (b:Entity {uuid: $root + '-b', name: 'Root B', group_id: $gid}),
               (:Episodic {uuid: $root + '-episode', name: 'user turn',
                           group_id: $gid, content: 'The root source sentence',
                           entity_edges: [$root]}),
               (a)-[:RELATES_TO {uuid: $root, group_id: $gid,
                                 fact: 'The root source sentence', name: 'rel',
                                 status: 'active', scope: 'real:global',
                                 episodes: [$root + '-episode'], created_at: $now}]->(b)
        """,
        gid=group_id,
        root=root,
        now=datetime.now(timezone.utc).isoformat(),
    )


async def land(
    driver: AutoGPTFalkorDriver,
    group_id: str,
    *,
    episode: str,
    name: str,
    edge: str,
    content: str,
    missing: tuple[str, ...] = (),
) -> None:
    """A write landing as graphiti saves one: its episode saved over by
    uuid (every property replaced), then the fact it produced. ``missing``
    are facts the episode lists that were never saved."""
    await driver.execute_query(
        """
        MERGE (ep:Episodic {uuid: $episode})
        SET ep = {uuid: $episode, name: $name, group_id: $gid, content: $content,
                  source: 'json', source_description: 'dream-pass consolidation',
                  entity_edges: [$edge] + $missing}
        CREATE (a:Entity {uuid: $edge + '-a', name: $edge + ' A', group_id: $gid}),
               (b:Entity {uuid: $edge + '-b', name: $edge + ' B', group_id: $gid}),
               (a)-[:RELATES_TO {uuid: $edge, group_id: $gid, fact: $content,
                                 name: 'derived', status: 'active',
                                 scope: 'real:global', episodes: [$episode],
                                 created_at: $now, fact_embedding: [0.1, 0.2]}]->(b)
        """,
        gid=group_id,
        episode=episode,
        name=name,
        edge=edge,
        content=content,
        missing=list(missing),
        now=datetime.now(timezone.utc).isoformat(),
    )


def placed_payload(name: str) -> dict[str, Any]:
    """What ``marked_write.placed`` is handed for a dream write ``name``."""
    return {
        "name": name,
        "episode_body": '{"content": "A dream fact"}',
        "source": EpisodeType.json,
        "source_description": "dream-pass consolidation",
        "reference_time": datetime.now(timezone.utc),
    }


async def markers(driver: AutoGPTFalkorDriver) -> list[dict[str, Any]]:
    """Every marker's uuid and state."""
    return await rows(
        driver,
        f"MATCH (m:{MARKER_LABEL}) RETURN m.uuid AS uuid, m.state AS state "
        "ORDER BY uuid",
    )


async def age(driver: AutoGPTFalkorDriver, marker: str) -> None:
    """Make ``marker`` look written long ago."""
    await driver.execute_query(
        f"MATCH (m:{MARKER_LABEL} {{uuid: $uuid}}) SET m.created_at = $long_ago",
        uuid=marker,
        long_ago=LONG_AGO,
    )


@contextmanager
def forget_mid_save(
    scope: MemoryScope,
    root: str,
    hard: bool,
    unfence: Callable[[], Awaitable[None]],
) -> Iterator[list[ForgetResult]]:
    """Forget ``root`` to the end right before graphiti saves the dream
    write's facts, once ``unfence`` has taken the lock's protection away."""
    original = FalkorDriverSession.run
    forgotten: list[ForgetResult] = []

    async def run(session: FalkorDriverSession, query: Any, **params: Any) -> Any:
        text = query if isinstance(query, str) else " ".join(q for q, _ in query)
        if not forgotten and "SET r = edge" in text:
            await unfence()
            forgotten.append(await retract(scope, [root], hard=hard))
        return await original(session, query, **params)

    with patch.object(FalkorDriverSession, "run", run):
        yield forgotten
