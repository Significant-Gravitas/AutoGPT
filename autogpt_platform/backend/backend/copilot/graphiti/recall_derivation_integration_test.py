"""``recall_derivation.STAMP_FACTS_QUERY`` on a live FalkorDB: of the facts a
dream write produced or merged into, it stamps only the live ones whose every
source is a recorded episode, each with the union of its sources' records.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_derivation_integration_test.py
"""

import pytest

from .falkordb_driver import AutoGPTFalkorDriver
from .recall_derivation import STAMP_FACTS_QUERY
from .recall_integration_fixtures import rows

_FACTS = [
    "from-one-dream",
    "from-both-dreams",
    "also-a-user-turn",
    "source-gone",
    "forgotten",
]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_only_facts_whose_sources_are_all_recorded_episodes_are_stamped(
    clean_graph: tuple[AutoGPTFalkorDriver, str],
) -> None:
    driver, group_id = clean_graph
    await _seed(driver, group_id)

    stamped = await rows(driver, STAMP_FACTS_QUERY, uuids=_FACTS, group_id=group_id)

    assert sorted(row["uuid"] for row in stamped) == [
        "from-both-dreams",
        "from-one-dream",
    ]
    records = await _records(driver)
    assert records["from-one-dream"] == (["fact-a", "fact-b"], ["turn-1"])
    facts, episodes = records["from-both-dreams"]
    assert sorted(facts) == ["fact-a", "fact-b", "fact-c"], "the union, once each"
    assert sorted(episodes) == ["turn-1", "turn-2"]
    for unstamped in ("also-a-user-turn", "source-gone", "forgotten"):
        assert records[unstamped] == (None, None), unstamped


async def _seed(driver: AutoGPTFalkorDriver, group_id: str) -> None:
    """Two recorded dream episodes, a user's turn with no record, and a fact
    for each case, one of them naming an episode the graph no longer has."""
    await driver.execute_query(
        """
        CREATE
          (a:Entity {uuid: 'a', group_id: $gid}),
          (b:Entity {uuid: 'b', group_id: $gid}),
          (:Episodic {uuid: 'dream-1', group_id: $gid,
                      derived_from_facts: ['fact-a', 'fact-b'],
                      derived_from_episodes: ['turn-1']}),
          (:Episodic {uuid: 'dream-2', group_id: $gid,
                      derived_from_facts: ['fact-b', 'fact-c'],
                      derived_from_episodes: ['turn-1', 'turn-2']}),
          (:Episodic {uuid: 'user-turn', group_id: $gid}),
          (a)-[:RELATES_TO {uuid: 'from-one-dream', group_id: $gid,
                            episodes: ['dream-1']}]->(b),
          (a)-[:RELATES_TO {uuid: 'from-both-dreams', group_id: $gid,
                            episodes: ['dream-1', 'dream-2']}]->(b),
          (a)-[:RELATES_TO {uuid: 'also-a-user-turn', group_id: $gid,
                            episodes: ['dream-1', 'user-turn']}]->(b),
          (a)-[:RELATES_TO {uuid: 'source-gone', group_id: $gid,
                            episodes: ['dream-1', 'deleted-turn']}]->(b),
          (a)-[:RELATES_TO {uuid: 'forgotten', group_id: $gid,
                            episodes: ['dream-1'],
                            forgotten_at: '2026-10-01T00:00:00+00:00'}]->(b)
        """,
        gid=group_id,
    )


async def _records(
    driver: AutoGPTFalkorDriver,
) -> dict[str, tuple[list[str] | None, list[str] | None]]:
    found = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO]->() "
        "RETURN e.uuid AS uuid, e.derived_from_facts AS facts, "
        "e.derived_from_episodes AS episodes",
    )
    return {row["uuid"]: (row["facts"], row["episodes"]) for row in found}
