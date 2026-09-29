"""Seeding and reads for the live dream-marker tests
(``recall_marker_integration_test.py``,
``recall_marker_crash_integration_test.py``): a user's fact and the
episode that stated it, a write landing as graphiti saves one, and the
markers a graph holds. Not collected by pytest.
"""

from datetime import datetime, timezone
from typing import Any

from graphiti_core.nodes import EpisodeType

from .falkordb_driver import AutoGPTFalkorDriver
from .recall_derivation import MARKER_LABEL
from .recall_integration_fixtures import rows

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
