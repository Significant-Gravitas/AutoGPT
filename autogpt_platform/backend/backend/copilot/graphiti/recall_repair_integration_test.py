"""Repairing forgotten edges graphiti merged a restated fact into, through
the production worker against a live FalkorDB: several facts at once, a
repair that fails, and a writer that changes the edge meanwhile.

Reproduced first by an independent validation
(``r3-ingestion-independent.py``: ``multi_merge``, ``repair_failure`` and
``concurrent_snapshot``). The unit siblings are ``recall_restate_test.py``
and ``recall_restore_test.py``.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_repair_integration_test.py
"""

import re
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from . import recall_restore
from .falkordb_driver import AutoGPTFalkorDriver
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BOB,
    BuildClient,
    Fact,
    edge_row,
    ingest_through_the_worker,
    live_facts,
    patch_recall_boundaries,
    recalled_episodes,
    rows,
    scripted_responses,
    stop_ingestion_workers,
)
from .recall_stash import read_forgets
from .scope import MemoryScope

_BUDGET: Fact = ("Alice", "Atlas", "Alice owns the Atlas budget")
_ASSIGNED: Fact = ("Alice", "Atlas", "Alice is assigned to work on the Atlas project")
_MANAGES: Fact = ("Alice", "Atlas", "Alice manages the Atlas budget")
_MERGE_INTO_FIRST = {"duplicate_facts": [0], "contradicted_facts": []}


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest_asyncio.fixture(loop_scope="function")
async def ingest_worker_cleanup(mocker: MockerFixture) -> AsyncIterator[None]:
    mocker.patch(
        "backend.copilot.dream.scheduling.ensure_dream_system_scheduled",
        AsyncMock(return_value=None),
    )
    yield
    await stop_ingestion_workers()


def _responses(facts: list[Fact], valid_at: str) -> dict[str, dict]:
    responses = scripted_responses(facts)
    for edge in responses["ExtractedEdges"]["edges"]:
        edge["valid_at"] = valid_at
    return responses


async def _learn_then_forget(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    build: BuildClient,
    facts: list[Fact],
) -> dict[str, dict[str, Any]]:
    """``facts`` through the worker, then forgotten: each edge's audit row."""
    client = build(driver, _responses(facts, "2026-01-01T00:00:00Z"))
    await ingest_through_the_worker(driver, scope, client, facts, session_id="s-first")
    forgotten = list(await live_facts(driver))
    await retract(scope, forgotten)
    return {uuid: await edge_row(driver, uuid) for uuid in forgotten}


async def _say_again(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    build: BuildClient,
    facts: list[Fact],
    merge: dict[str, list[int]] | None = None,
) -> str:
    responses = _responses(facts, "2026-06-01T00:00:00Z")
    client = build(driver, responses)
    answer = client.llm_client._generate_response

    async def resolve(messages, response_model=None, *args: Any, **kwargs: Any):
        if response_model is not None and response_model.__name__ == "EdgeDuplicate":
            return merge or _merge_by_topic(messages[-1].content)
        return await answer(messages, response_model, *args, **kwargs)

    with patch.object(client.llm_client, "_generate_response", side_effect=resolve):
        return await ingest_through_the_worker(
            driver, scope, client, facts, session_id="s-again"
        )


def _merge_by_topic(prompt: str) -> dict[str, list[int]]:
    """graphiti's model calls each restatement a duplicate of a different
    forgotten edge (as ``r3-ingestion-independent.py`` scripted it)."""
    found = re.search(r"<NEW FACT>\s*(.*?)\s*</NEW FACT>", prompt, re.DOTALL)
    budget = found is not None and "budget" in found.group(1)
    return {"duplicate_facts": [1 if budget else 0], "contradicted_facts": []}


@contextmanager
def _restores_fail() -> Iterator[None]:
    original = AutoGPTFalkorDriver.execute_query

    async def execute(self, cypher_query_, **params):
        if cypher_query_ == recall_restore._RESTORE_QUERY:
            raise RuntimeError("restore lost")
        return await original(self, cypher_query_, **params)

    with patch.object(AutoGPTFalkorDriver, "execute_query", execute):
        yield


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_facts_said_again_between_the_same_entities_both_come_back(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    driver, scope = scope_graph
    audits = await _learn_then_forget(
        driver, scope, stub_graphiti_client, [ALICE, _BUDGET]
    )

    episode = await _say_again(
        driver, scope, stub_graphiti_client, [_ASSIGNED, _MANAGES]
    )

    live = await live_facts(driver)
    assert sorted(live.values()) == sorted([_ASSIGNED[2], _MANAGES[2]])
    for uuid, audit in audits.items():
        assert await edge_row(driver, uuid) == audit, "a forgotten edge changed"
    assert episode in await recalled_episodes(scope)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("finisher", ["next-ingestion", "forget-again"])
async def test_a_repair_that_failed_is_finished_later(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, finisher: str
) -> None:
    """Both tries of the restore fail: the fact said again is live all the
    same, and the forgotten edge's own fields come back from the stash."""
    driver, scope = scope_graph
    [(alice, audit)] = (
        await _learn_then_forget(driver, scope, stub_graphiti_client, [ALICE])
    ).items()

    with _restores_fail():
        again = await _say_again(
            driver, scope, stub_graphiti_client, [_ASSIGNED], _MERGE_INTO_FIRST
        )

    assert list((await live_facts(driver)).values()) == [_ASSIGNED[2]]
    assert again in await recalled_episodes(scope)
    assert (await read_forgets(scope.group_id))[alice].dropped_episodes == [again]
    if finisher == "next-ingestion":
        client = stub_graphiti_client(driver, scripted_responses([BOB]))
        await ingest_through_the_worker(driver, scope, client, [BOB], session_id="s-3")
    else:
        await retract(scope, [alice])
    assert await edge_row(driver, alice) == audit
    assert again in await recalled_episodes(scope)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_repair_keeps_what_another_writer_changed(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    """Between graphiti's write and the repair another writer corrects the
    edge's confidence and provenance and adds a source: the repair sets back
    only what the forget owns."""
    driver, scope = scope_graph
    [(alice, audit)] = (
        await _learn_then_forget(driver, scope, stub_graphiti_client, [ALICE])
    ).items()
    original = AutoGPTFalkorDriver.execute_query

    async def execute(self, cypher_query_, **params):
        if cypher_query_ == recall_restore._RESTORE_QUERY:
            await original(self, _CORRECTION, uuid=alice)
        return await original(self, cypher_query_, **params)

    with patch.object(AutoGPTFalkorDriver, "execute_query", execute):
        await _say_again(
            driver, scope, stub_graphiti_client, [_ASSIGNED], _MERGE_INTO_FIRST
        )

    [edge] = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() RETURN properties(e) AS p",
        uuid=alice,
    )
    kept = edge["p"]
    assert (kept["confidence"], kept["provenance"]) == (0.92, "audit-correction")
    assert kept["episodes"] == [*audit["episodes"], "concurrent-source"]
    row = await edge_row(driver, alice)
    assert {k: v for k, v in row.items() if k != "episodes"} == {
        k: v for k, v in audit.items() if k != "episodes"
    }


_CORRECTION = """
MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
SET e.confidence = 0.92,
    e.provenance = 'audit-correction',
    e.episodes = coalesce(e.episodes, []) + ['concurrent-source']
"""
