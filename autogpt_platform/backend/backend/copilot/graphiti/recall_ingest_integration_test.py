"""A forgotten fact stated again, through the production ingestion worker
against a live FalkorDB.

graphiti resolves a newly extracted fact against every edge between the same
entities, the forgotten one included. Before a forget scrubbed the edge's
sentence, its exact-text match merged the new statement into the retracted
edge: the user got no live fact and the new episode was hidden with the old
one. The scrub stops that match; graphiti's model can still call the new fact
a duplicate of the forgotten edge, or a contradiction of it, and
``recall_ingest`` puts the edge back and gives the fact a new live edge.
The unit sibling is ``recall_ingest_test.py``.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_ingest_integration_test.py
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from .falkordb_driver import AutoGPTFalkorDriver
from .recall import FORGOTTEN_FACT
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BuildClient,
    Fact,
    edge_row,
    episode_row,
    ingest_through_the_worker,
    live_facts,
    patch_recall_boundaries,
    recalled_episodes,
    recalled_facts,
    scripted_responses,
    stop_ingestion_workers,
)
from .scope import MemoryScope

_REWORDED: Fact = ("Alice", "Atlas", "Alice leads the Atlas project")
# What graphiti's model may answer about the one existing edge between Alice
# and Atlas (index 0): the same fact, or one the new fact contradicts.
_RESOLUTIONS = {
    "duplicate": {"duplicate_facts": [0], "contradicted_facts": []},
    "contradiction": {"duplicate_facts": [], "contradicted_facts": [0]},
}
# When each statement says its fact became true: graphiti only lets a newer
# fact invalidate an older one.
_FIRST_VALID_AT = "2026-01-01T00:00:00Z"
_AGAIN_VALID_AT = "2026-06-01T00:00:00Z"


def _responses(
    fact: Fact, valid_at: str, resolution: dict[str, list[int]] | None = None
) -> dict[str, dict]:
    responses = scripted_responses([fact])
    responses["ExtractedEdges"]["edges"][0]["valid_at"] = valid_at
    if resolution is not None:
        responses["EdgeDuplicate"] = resolution
    return responses


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest_asyncio.fixture(loop_scope="function")
async def ingest_worker_cleanup(mocker: MockerFixture) -> AsyncIterator[None]:
    """No dream registration on a first write; no idle worker left behind."""
    mocker.patch(
        "backend.copilot.dream.scheduling.ensure_dream_system_scheduled",
        AsyncMock(return_value=None),
    )
    yield
    await stop_ingestion_workers()


async def _learn_then_forget(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient
) -> tuple[str, dict[str, Any]]:
    """Alice's fact through the worker, forgotten: its uuid and audit row."""
    client = build(driver, _responses(ALICE, _FIRST_VALID_AT))
    await ingest_through_the_worker(
        driver, scope, client, [ALICE], session_id="s-first"
    )
    [alice] = await live_facts(driver)
    await retract(scope, [alice])
    return alice, await edge_row(driver, alice)


async def _state_again(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    build: BuildClient,
    fact: Fact,
    resolution: dict[str, list[int]] | None = None,
) -> tuple[str, list[str]]:
    """``fact`` stated in a new chat turn: the new episode's uuid and every
    message graphiti sent its model for it."""
    client = build(driver, _responses(fact, _AGAIN_VALID_AT, resolution))
    answer = client.llm_client._generate_response
    with patch.object(
        client.llm_client, "_generate_response", side_effect=answer
    ) as generate:
        episode = await ingest_through_the_worker(
            driver, scope, client, [fact], session_id="s-again"
        )
    sent = [m.content for call in generate.await_args_list for m in call.args[0]]
    return episode, sent


async def _assert_live_again(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    forgotten: tuple[str, dict[str, Any]],
    episode: str,
    sentence: str,
) -> None:
    """Exactly one live fact, the new one; the forget's audit edge as the
    forget left it; the new episode recallable and citing only the new fact."""
    alice, audit = forgotten
    live = await live_facts(driver)
    assert list(live.values()) == [sentence], "exactly one live fact"
    [restated] = live
    assert restated != alice, "the forgotten edge must stay forgotten"
    assert await edge_row(driver, alice) == audit, "the audit edge changed"
    assert await recalled_facts(scope) == {restated}
    assert episode in await recalled_episodes(scope)
    assert (await episode_row(driver, episode))["entity_edges"] == [restated]
    assert (await edge_row(driver, restated))["episodes"] == [episode]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_fact_taught_again_after_a_forget_is_live_again(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    driver, scope = scope_graph
    forgotten = await _learn_then_forget(driver, scope, stub_graphiti_client)

    episode, _ = await _state_again(driver, scope, stub_graphiti_client, ALICE)

    await _assert_live_again(driver, scope, forgotten, episode, ALICE[2])


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("resolution", list(_RESOLUTIONS))
async def test_a_fact_resolved_into_a_forgotten_one_gets_its_own_live_edge(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, resolution: str
) -> None:
    driver, scope = scope_graph
    forgotten = await _learn_then_forget(driver, scope, stub_graphiti_client)

    episode, sent = await _state_again(
        driver, scope, stub_graphiti_client, ALICE, _RESOLUTIONS[resolution]
    )

    assert any(FORGOTTEN_FACT in message for message in sent), "not offered"
    await _assert_live_again(driver, scope, forgotten, episode, ALICE[2])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_new_fact_merged_into_a_forgotten_one_never_shows_its_sentence(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    """graphiti's model calls a reworded fact a duplicate of the forgotten
    edge: the old sentence reaches none of the prompts that follow, the
    edge's attribute prompt included, and the new wording is the live fact."""
    driver, scope = scope_graph
    forgotten = await _learn_then_forget(driver, scope, stub_graphiti_client)

    episode, sent = await _state_again(
        driver, scope, stub_graphiti_client, _REWORDED, _RESOLUTIONS["duplicate"]
    )

    assert any(FORGOTTEN_FACT in message for message in sent), "not offered"
    assert [message for message in sent if ALICE[2] in message] == []
    await _assert_live_again(driver, scope, forgotten, episode, _REWORDED[2])
