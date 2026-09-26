"""The forget contract against a live FalkorDB.

A forget hides the fact and the text it came from from every read the
assistant or a dream makes, while the audit record stays: a hard forget never
deletes an episode a retained edge still cites. Each case was reproduced
against the first recall-policy commit by an independent validation; the
unit siblings (``recall_forget_test.py``, ``recall_test.py``) pin the Cypher.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_forget_integration_test.py
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from backend.copilot.dream import ratification
from backend.copilot.dream.fetch import _fetch_recent_episodes
from backend.copilot.model import ChatSession
from backend.copilot.tools import graphiti_search
from backend.copilot.tools.models import MemorySearchResponse

from . import context, recall
from .falkordb_driver import AutoGPTFalkorDriver
from .memory_model import MemoryForgetFailureCode
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BOB,
    CAROL,
    capture_spawned_tasks,
    edge_row,
    episode_row,
    ingest_facts,
    patch_recall_boundaries,
    recalled_facts,
)
from .scope import MemoryScope

_LONG_AGO = "2025-01-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest.fixture
def hit_tasks(mocker) -> list[asyncio.Task]:
    return capture_spawned_tasks(mocker)


async def _what_recall_shows(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, hit_tasks: list[asyncio.Task]
) -> list[str]:
    """Every episode text a reader gets: recent episodes, warm context,
    ``memory_search`` and the dream's episode gather."""
    recent = [episode.content for episode in await recall.recent_episodes(scope, 5)]
    warm = await context._fetch(scope, "Atlas")
    await asyncio.gather(*context._pending_hit_tasks)
    session = ChatSession.new(scope.owner_user_id, dry_run=False)
    found = await graphiti_search.MemorySearchTool()._execute(
        scope.owner_user_id, session, query="Atlas"
    )
    await asyncio.gather(*hit_tasks)
    assert isinstance(found, MemorySearchResponse)
    window_start = datetime.now(timezone.utc) - timedelta(days=14)
    dream = await _fetch_recent_episodes(driver, scope.group_id, window_start, 50)
    return [
        *recent,
        warm or "",
        *found.recent_episodes,
        *(episode.content or "" for episode in dream),
    ]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("hard", [False, True], ids=["soft", "hard"])
async def test_forgetting_a_fact_hides_the_episode_it_shares(
    scope_graph, stub_graphiti_client, hit_tasks, hard: bool
) -> None:
    driver, scope = scope_graph
    episode, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE, BOB]
    )

    result = await retract(scope, [edges[ALICE[2]]], hard=hard)

    assert result.deleted == [edges[ALICE[2]]] and result.failures == []
    shown = await _what_recall_shows(driver, scope, hit_tasks)
    assert not any(ALICE[2] in text for text in shown), "forgotten text recalled"
    assert await recalled_facts(scope) == {edges[BOB[2]]}, "Bob's fact stays"
    kept = await episode_row(driver, episode)
    assert kept["redacted_at"] is not None
    assert ALICE[2] in kept["content"], "the stored text is kept for audit"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_from_before_the_policy_still_hides_its_episode(
    scope_graph, stub_graphiti_client, hit_tasks
) -> None:
    """The pre-policy forget set ``expired_at`` and nothing else, and no
    episode was redacted; the read side must recognise that shape."""
    driver, scope = scope_graph
    _, edges = await ingest_facts(driver, scope, stub_graphiti_client, [ALICE])
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e.expired_at = $old",
        uuid=edges[ALICE[2]],
        old=_LONG_AGO,
    )

    shown = await _what_recall_shows(driver, scope, hit_tasks)

    assert not any(ALICE[2] in text for text in shown), "forgotten text recalled"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_failed_redaction_is_reported_and_the_text_stays_hidden(
    scope_graph, stub_graphiti_client, hit_tasks
) -> None:
    driver, scope = scope_graph
    episode, edges = await ingest_facts(driver, scope, stub_graphiti_client, [ALICE])
    write = AutoGPTFalkorDriver.execute_query

    async def lose_the_redaction(self, cypher_query_, **params):
        if "SET ep.redacted_at" in cypher_query_:
            raise RuntimeError("redaction write lost")
        return await write(self, cypher_query_, **params)

    with patch.object(AutoGPTFalkorDriver, "execute_query", lose_the_redaction):
        result = await retract(scope, [edges[ALICE[2]]])

    shown = await _what_recall_shows(driver, scope, hit_tasks)
    assert not any(ALICE[2] in text for text in shown), "forgotten text recalled"
    assert (await episode_row(driver, episode))["redacted_at"] is None
    assert result.deleted == [edges[ALICE[2]]], "the fact itself was forgotten"
    assert [(f.uuid, f.code) for f in result.failures] == [
        (edges[ALICE[2]], MemoryForgetFailureCode.CLEANUP_ERROR)
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hard_forget_keeps_an_episode_a_retracted_fact_still_cites(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    episode, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE, BOB]
    )
    alice, bob = edges[ALICE[2]], edges[BOB[2]]
    await retract(scope, [bob])

    result = await retract(scope, [alice], hard=True)

    assert result.deleted == [alice] and result.failures == []
    assert result.deleted_episodes == [], "Bob's retained edge still cites it"
    assert await edge_row(driver, alice) == {}
    retained = await edge_row(driver, bob)
    assert retained["status"] == "retracted"
    assert episode in retained["episodes"]
    assert (await episode_row(driver, episode))["entity_edges"] == [bob]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_keeps_the_expiry_and_valid_time_already_set(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    _, edges = await ingest_facts(driver, scope, stub_graphiti_client, [ALICE])
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() "
        "SET e.expired_at = $old, e.invalid_at = $old",
        uuid=edges[ALICE[2]],
        old=_LONG_AGO,
    )

    await retract(scope, [edges[ALICE[2]]])

    row = await edge_row(driver, edges[ALICE[2]])
    assert row["status"] == "retracted"
    assert row["expired_at"] == _LONG_AGO, "the first retirement time is kept"
    assert row["invalid_at"] == _LONG_AGO


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("hits", [2, 0], ids=["promote", "supersede"])
async def test_the_ratification_sweep_never_overwrites_a_forget(
    scope_graph, stub_graphiti_client, mocker, hits: int
) -> None:
    """The user forgets a tentative fact after the nightly sweep listed it
    and before the sweep writes: neither the promotion nor the unratified
    supersession may replace the retraction."""
    driver, scope = scope_graph
    _, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [CAROL], status="tentative"
    )
    carol = edges[CAROL[2]]
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e.created_at = $old",
        uuid=carol,
        old=_LONG_AGO,  # past the grace period, so no hits means supersede
    )

    async def forget_then_count(_scope: MemoryScope, edge_uuid: str) -> int:
        await retract(scope, [edge_uuid])
        return hits

    mocker.patch.object(ratification, "_get_hit_count", forget_then_count)
    result = await ratification.run_ratification_pass(scope.owner_user_id)

    assert result.examined_count == 1
    assert (result.ratified_count, result.superseded_count) == (0, 0)
    row = await edge_row(driver, carol)
    assert (row["status"], row["reason"]) == ("retracted", "user_signal")
