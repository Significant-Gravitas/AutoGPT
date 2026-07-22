from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from .lifecycle import filter_active_shared_edges


@pytest.mark.integration
@pytest.mark.asyncio
async def test_shared_lifecycle_filter_enforces_active_and_unexpired(
    seeded_graph,
) -> None:
    driver, group_id = seeded_graph
    now = datetime.now(timezone.utc).isoformat()
    await driver.execute_query(
        """
        MATCH (a:Entity {uuid: 'alice'}), (t:Entity {uuid: 'atlas'})
        SET a.group_id = $group_id,
            t.group_id = $group_id
        CREATE
          (a)-[:RELATES_TO {
            uuid: 'e3', group_id: $group_id, status: 'superseded',
            expired_at: $now, fact: 'superseded'
          }]->(t),
          (a)-[:RELATES_TO {
            uuid: 'e4', group_id: $group_id, status: 'active',
            expired_at: $now, fact: 'expired active'
          }]->(t)
        WITH a
        MATCH ()-[e:RELATES_TO {uuid: 'e2'}]->()
        SET e.status = 'tentative'
        """,
        group_id=group_id,
        now=now,
    )
    edges = [SimpleNamespace(uuid=f"e{number}") for number in range(1, 5)]

    result = await filter_active_shared_edges(group_id, edges, driver=driver)

    assert [edge.uuid for edge in result] == ["e1"]
