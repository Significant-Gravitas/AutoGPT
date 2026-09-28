"""The dream's writers against a user's forget, on a live FalkorDB.

A dream reads the graph, asks a model, then writes: supersessions
(``mark_edges_superseded``) and single-hop neighbour invalidations
(``invalidate_entity_direct_neighbors``). A user can forget a fact in
between. Both writers only write over a live fact, and a forget's
``forgotten_at`` is written by nothing but ``recall_forget``, so wherever the
dream's write lands (between the forget's edge write and the rest of the
forget, after a whole forget, after a forget whose clean-up failed, or on a
forget from before the recall policy) the fact stays forgotten, its episode
and chat session stay hidden, and the forget's audit fields stay as it left
them. Each case was reproduced against the previous commit by an independent
validation (``r2-dream-atomicity.py``).

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_dream_writers_integration_test.py
"""

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.dream import demotions
from backend.copilot.dream.fetch import _fetch_recent_episodes
from backend.copilot.dream.hidden_sessions import hidden_session_ids
from backend.copilot.dream.schemas import (
    DreamDemotion,
    DreamOperations,
    EntityInvalidation,
)

from . import recall_hide
from .falkordb_driver import AutoGPTFalkorDriver
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    edge_row,
    ingest_facts,
    patch_recall_boundaries,
    recalled_episodes,
    rows,
)
from .scope import MemoryScope

_LONG_AGO = "2025-01-01T00:00:00+00:00"
_WRITERS = ["supersede", "invalidate-neighbours"]
_TIMINGS = ["mid-forget", "after-forget", "after-failed-clean-up", "legacy-forget"]


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


async def _dream_write(
    scope: MemoryScope, writer: str, edge_uuid: str, entity_uuid: str
) -> int:
    """The dream's write, as its destructive stage (``demotions.py``) makes
    it; how many edges it changed."""
    if writer == "supersede":
        ops = DreamOperations(
            demotions=[DreamDemotion(edge_uuid=edge_uuid, reason="stale_fact")]
        )
    else:
        ops = DreamOperations(
            entity_invalidations=[
                EntityInvalidation(entity_uuid=entity_uuid, reason="dead_client")
            ]
        )
    with patch.object(demotions, "is_feature_enabled", AsyncMock(return_value=True)):
        results = await demotions.apply_demotions(scope, "p-writers", ops, {edge_uuid})
    return results.demoted + results.entity_edges


async def _forget(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    edge_uuid: str,
    timing: str,
    dream_write: Callable[[], Awaitable[None]],
) -> None:
    """Forget ``edge_uuid`` with the dream's write landing at ``timing``."""
    if timing == "legacy-forget":
        await driver.execute_query(
            "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e.expired_at = $old",
            uuid=edge_uuid,
            old=_LONG_AGO,
        )
    elif timing == "after-forget":
        await retract(scope, [edge_uuid])
    else:
        await _forget_with_the_write_inside(scope, edge_uuid, timing, dream_write)
    if timing != "mid-forget":
        await dream_write()


async def _forget_with_the_write_inside(
    scope: MemoryScope,
    edge_uuid: str,
    timing: str,
    dream_write: Callable[[], Awaitable[None]],
) -> None:
    """Right after the forget's edge write, before any of its clean-up: the
    dream writes there, or the clean-up fails there."""
    original = AutoGPTFalkorDriver.execute_query

    async def interleave(self, cypher_query_, **params):
        if cypher_query_ == recall_hide.SCRUB_FACTS_QUERY:
            if timing == "after-failed-clean-up":
                raise RuntimeError("clean-up lost")
            await dream_write()
        return await original(self, cypher_query_, **params)

    with patch.object(AutoGPTFalkorDriver, "execute_query", interleave):
        await retract(scope, [edge_uuid])


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("timing", _TIMINGS)
@pytest.mark.parametrize("writer", _WRITERS)
async def test_a_dream_write_never_undoes_a_forget(
    scope_graph, stub_graphiti_client, writer: str, timing: str
) -> None:
    driver, scope = scope_graph
    episode, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE], session_id="s-forgotten"
    )
    alice = edges[ALICE[2]]
    [source] = await rows(
        driver,
        "MATCH (a)-[e:RELATES_TO {uuid: $uuid}]->() RETURN a.uuid AS uuid",
        uuid=alice,
    )
    written: list[int] = []

    async def dream_write() -> None:
        written.append(await _dream_write(scope, writer, alice, source["uuid"]))

    await _forget(driver, scope, alice, timing, dream_write)

    assert written == [0], "the dream wrote over a forgotten fact"
    row = await edge_row(driver, alice)
    if timing == "legacy-forget":
        assert (row["status"], row["reason"]) == ("active", None)
        assert (row["expired_at"], row["forgotten_at"]) == (_LONG_AGO, None)
    else:
        assert (row["status"], row["reason"]) == ("retracted", "user_signal")
        assert row["forgotten_at"] is not None
    assert row["invalid_at"] is None, "a forget is not a world change"
    window_start = datetime.now(timezone.utc) - timedelta(days=14)
    dream = await _fetch_recent_episodes(driver, scope.group_id, window_start, 50)
    assert episode not in {e.uuid for e in dream}, "the dream reads it again"
    assert episode not in await recalled_episodes(scope)
    assert await hidden_session_ids(driver, scope.group_id) == {"s-forgotten"}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("writer", _WRITERS)
async def test_the_dream_still_retires_a_live_fact(
    scope_graph, stub_graphiti_client, writer: str
) -> None:
    driver, scope = scope_graph
    _, edges = await ingest_facts(driver, scope, stub_graphiti_client, [ALICE])
    alice = edges[ALICE[2]]
    [source] = await rows(
        driver,
        "MATCH (a)-[e:RELATES_TO {uuid: $uuid}]->() RETURN a.uuid AS uuid",
        uuid=alice,
    )

    assert await _dream_write(scope, writer, alice, source["uuid"]) == 1

    row = await edge_row(driver, alice)
    assert row["status"] == "superseded" and row["expired_at"] is not None
    assert row["forgotten_at"] is None, "only a forget marks a fact forgotten"
