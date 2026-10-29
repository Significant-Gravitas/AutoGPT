"""Paging the settings fact list against a live FalkorDB (``fact_list.py``).

An account with 19 memories pages through all of them, newest first, each
exactly once, with the total on every page; facts that are not live are
neither listed nor counted; and a fact forgotten while someone pages does not
push another fact out of the next page.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/fact_list_integration_test.py
"""

from datetime import UTC, datetime, timedelta

import pytest

from .fact_list import list_live_facts
from .falkordb_driver import AutoGPTFalkorDriver
from .scope import MemoryScope

_START = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


async def _seed(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    count: int,
    *,
    same_time: bool = False,
    extra: str = "",
) -> list[str]:
    """``count`` facts from one user entity, newest first in the returned
    order (by ``created_at``, then uuid for facts stored at the same time)."""
    uuids = [f"fact-{i:02d}" for i in range(count)]
    for i, uuid in enumerate(uuids):
        at = _START if same_time else _START - timedelta(minutes=i)
        await driver.execute_query(
            f"""
            MERGE (u:Entity {{uuid: 'user', name: 'User', group_id: $g}})
            CREATE (t:Entity {{uuid: $t, name: $name, group_id: $g}})
            CREATE (u)-[:RELATES_TO {{uuid: $e, group_id: $g, fact: $fact,
                name: 'knows', created_at: $at{extra}}}]->(t)
            """,
            g=scope.group_id,
            t=f"target-{uuid}",
            name=f"Thing {i}",
            e=uuid,
            fact=f"Fact number {i}",
            at=at.isoformat(),
        )
    return sorted(uuids, reverse=True) if same_time else uuids


async def _page_through(scope: MemoryScope, limit: int) -> tuple[list[str], set]:
    uuids: list[str] = []
    totals: set[int] = set()
    cursor = None
    while True:
        page = await list_live_facts(scope, limit=limit, cursor=cursor)
        uuids += [fact.uuid for fact in page.items]
        totals.add(page.total)
        if page.next_cursor is None:
            return uuids, totals
        cursor = page.next_cursor


@pytest.mark.integration
async def test_pages_through_every_live_fact_once_with_the_total(
    scope_graph: tuple[AutoGPTFalkorDriver, MemoryScope],
) -> None:
    driver, scope = scope_graph
    expected = await _seed(driver, scope, 19)

    first = await list_live_facts(scope, limit=6)
    assert [fact.uuid for fact in first.items] == expected[:6]
    assert first.total == 19
    assert first.items[0].fact == "Fact number 0"
    assert first.items[0].source == "User"
    assert first.items[0].target == "Thing 0"

    uuids, totals = await _page_through(scope, limit=6)
    assert uuids == expected
    assert totals == {19}


@pytest.mark.integration
async def test_facts_stored_at_the_same_time_are_each_listed_once(
    scope_graph: tuple[AutoGPTFalkorDriver, MemoryScope],
) -> None:
    driver, scope = scope_graph
    expected = await _seed(driver, scope, 7, same_time=True)

    uuids, totals = await _page_through(scope, limit=2)

    assert uuids == expected
    assert totals == {7}


@pytest.mark.integration
async def test_facts_that_are_not_live_are_neither_listed_nor_counted(
    scope_graph: tuple[AutoGPTFalkorDriver, MemoryScope],
) -> None:
    driver, scope = scope_graph
    live = await _seed(driver, scope, 3)
    await driver.execute_query(
        """
        MATCH (u:Entity {uuid: 'user'})
        CREATE (u)-[:RELATES_TO {uuid: 'forgotten', group_id: $g, fact: 'x',
            created_at: $at, forgotten_at: $at, status: 'retracted'}]->
            (:Entity {uuid: 't-forgotten', name: 'F', group_id: $g})
        CREATE (u)-[:RELATES_TO {uuid: 'expired', group_id: $g, fact: 'y',
            created_at: $at, expired_at: $at}]->
            (:Entity {uuid: 't-expired', name: 'E', group_id: $g})
        """,
        g=scope.group_id,
        at=(_START + timedelta(days=1)).isoformat(),
    )

    page = await list_live_facts(scope, limit=50)

    assert [fact.uuid for fact in page.items] == live
    assert page.total == 3
    assert page.next_cursor is None


@pytest.mark.integration
async def test_a_fact_forgotten_between_pages_does_not_skip_the_next_one(
    scope_graph: tuple[AutoGPTFalkorDriver, MemoryScope],
) -> None:
    """With an offset, forgetting a fact on the first page would shift the
    first fact of the second page onto the first, out of sight."""
    driver, scope = scope_graph
    expected = await _seed(driver, scope, 12)

    first = await list_live_facts(scope, limit=6)
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $u}]->() SET e.forgotten_at = $at",
        u=first.items[0].uuid,
        at=_START.isoformat(),
    )
    second = await list_live_facts(scope, limit=6, cursor=first.next_cursor)

    assert [fact.uuid for fact in second.items] == expected[6:]
    assert second.total == 11
    assert second.next_cursor is None


@pytest.mark.integration
async def test_a_scope_without_a_graph_has_no_facts(
    scope_graph: tuple[AutoGPTFalkorDriver, MemoryScope],
) -> None:
    _, scope = scope_graph
    empty = MemoryScope.for_user(f"{scope.owner_user_id}-none")

    page = await list_live_facts(empty, limit=6)

    assert page.items == []
    assert page.total == 0
    assert page.next_cursor is None
