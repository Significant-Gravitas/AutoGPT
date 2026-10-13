"""A statement merged into a stored fact leaves every attribute on it as it
was (``fact_attributes.py``), through the production worker against a live
FalkorDB. That covers the platform's stamps, the dream's envelope metadata
and the status, whatever graphiti's model answers, on both of graphiti's
merge branches and its exact-text path. A forgotten fact is still no merge
target, and FalkorDB's ``+=`` behaves as the merging save relies on.

Reproduced first by a probe of the same shape before the fix: every stamp
was lost on a merge, and the status was the model's answer.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/fact_attributes_integration_test.py
"""

import ast
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
    edge_properties,
    ingest_through_the_worker,
    live_facts,
    model_of,
    patch_recall_boundaries,
    scripted_responses,
    stop_ingestion_workers,
)
from .scope import MemoryScope

_PARAPHRASE: Fact = ("Alice", "Atlas", "Alice is working on Atlas")
_TENTATIVE = {"status": "tentative", "scope": "real:global"}

# What the platform writes on a live fact outside graphiti's extraction: the
# recall stamps, the ratification and web-verification stamps, and the
# dream's envelope metadata (``ingest._stamp_edge_metadata``).
STAMPS: dict[str, Any] = {
    "recall_count": 3,
    "last_recalled_at": "2026-05-01T00:00:00.000000+00:00",
    "prev_recalled_at": "2026-04-01T00:00:00.000000+00:00",
    "ratified_at": "2026-05-02T00:00:00+00:00",
    "web_verified_at": "2026-05-03T00:00:00+00:00",
    "status": "active",
    "source_kind": "assistant_derived",
    "confidence": 0.9,
    "provenance": "dream:pass-1",
    "scope": "project:atlas",
}


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest_asyncio.fixture(loop_scope="function")
async def ingest_worker_cleanup(mocker: MockerFixture) -> AsyncIterator[None]:
    mocker.patch(
        "backend.copilot.dream.registry.ensure_dream_system_scheduled",
        AsyncMock(return_value=None),
    )
    yield
    await stop_ingestion_workers()


def _responses(fact: Fact, relation: str, valid_at: str) -> dict[str, dict]:
    responses = scripted_responses([fact])
    for edge in responses["ExtractedEdges"]["edges"]:
        edge["valid_at"] = valid_at
        edge["relation_type"] = relation
    return responses


async def _stamped_fact(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient, relation: str
) -> str:
    """``ALICE`` learnt through the worker as a ``relation`` edge, then
    stamped with every one of ``STAMPS``."""
    client = build(driver, _responses(ALICE, relation, "2026-01-01T00:00:00Z"))
    await ingest_through_the_worker(driver, scope, client, [ALICE], session_id="s-1")
    [uuid] = list(await live_facts(driver))
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e += $stamps",
        uuid=uuid,
        stamps=STAMPS,
    )
    return uuid


async def _restate(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    build: BuildClient,
    fact: Fact,
    *,
    duplicate_of: str,
) -> tuple[str, list[str]]:
    """``fact`` in a new turn, graphiti's model naming the candidate whose
    text is ``duplicate_of`` a duplicate and answering ``_TENTATIVE`` for its
    attributes: the new episode and the models asked."""
    responses = _responses(fact, "MemoryFact", "2026-06-01T00:00:00Z")
    responses["MemoryFact"] = dict(_TENTATIVE)
    client = build(driver, responses)
    model = model_of(client)
    scripted = model._generate_response

    async def resolve(messages, response_model=None, *args: Any, **kwargs: Any):
        if response_model is not None and response_model.__name__ == "EdgeDuplicate":
            listed = _existing_facts(messages[-1].content)
            same = [c["idx"] for c in listed if c["fact"] == duplicate_of]
            return {"duplicate_facts": same[:1], "contradicted_facts": []}
        return await scripted(messages, response_model, *args, **kwargs)

    with patch.object(model, "_generate_response", side_effect=resolve) as calls:
        episode = await ingest_through_the_worker(
            driver, scope, client, [fact], session_id="s-2"
        )
    return episode, [call.args[1].__name__ for call in calls.await_args_list]


def _existing_facts(prompt: str) -> list[dict[str, Any]]:
    start = prompt.index("<EXISTING FACTS>") + len("<EXISTING FACTS>")
    return ast.literal_eval(prompt[start : prompt.index("</EXISTING FACTS>")].strip())


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relation",
    ["MemoryFact", "WORKS_ON"],
    ids=["named-after-its-edge-type", "named-otherwise"],
)
async def test_a_merged_restatement_leaves_every_attribute_as_it_was(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, relation: str
) -> None:
    """graphiti's model merges the restatement into the stored fact and
    answers ``tentative`` for it: the active fact stays active, and keeps
    its scope and every stamp. graphiti asked the model for nothing: on the
    first branch the client answered for it, on the second graphiti asks no
    attributes of an edge whose name is no edge type."""
    driver, scope = scope_graph
    uuid = await _stamped_fact(driver, scope, stub_graphiti_client, relation)
    before = await edge_properties(driver, uuid)

    episode, asked = await _restate(
        driver, scope, stub_graphiti_client, _PARAPHRASE, duplicate_of=ALICE[2]
    )

    after = await edge_properties(driver, uuid)
    assert episode in after["episodes"], "the restatement was merged into it"
    assert list(await live_facts(driver)) == [uuid], "and saved nowhere else"
    assert {key: after.get(key) for key in STAMPS} == STAMPS
    assert after["status"] == "active", "the model's tentative was not taken"
    assert {k: v for k, v in after.items() if k != "episodes"} == {
        k: v for k, v in before.items() if k != "episodes"
    }
    assert "MemoryFact" not in asked


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_same_sentence_again_leaves_every_attribute_as_it_was(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    """graphiti's exact-text match, which runs before its model."""
    driver, scope = scope_graph
    uuid = await _stamped_fact(driver, scope, stub_graphiti_client, "MemoryFact")

    episode, asked = await _restate(
        driver, scope, stub_graphiti_client, ALICE, duplicate_of=ALICE[2]
    )

    after = await edge_properties(driver, uuid)
    assert episode in after["episodes"]
    assert {key: after.get(key) for key in STAMPS} == STAMPS
    assert "EdgeDuplicate" not in asked and "MemoryFact" not in asked


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forgotten_fact_is_no_merge_target(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    """graphiti's model names the forgotten fact a duplicate: the statement
    is saved as a new live fact and the forgotten one does not change."""
    driver, scope = scope_graph
    uuid = await _stamped_fact(driver, scope, stub_graphiti_client, "MemoryFact")
    await retract(scope, [uuid])
    before = await edge_properties(driver, uuid)

    await _restate(
        driver, scope, stub_graphiti_client, _PARAPHRASE, duplicate_of=FORGOTTEN_FACT
    )

    assert await edge_properties(driver, uuid) == before
    assert sorted((await live_facts(driver)).values()) == [_PARAPHRASE[2]]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_falkordb_merges_a_map_into_an_edge(scope_graph) -> None:
    """What the merging save relies on, in both of graphiti's save forms: a
    property the map does not name stays, a null in the map removes one, and
    the rest are set; ``=`` instead removes every property the map does not
    name."""
    driver, _ = scope_graph
    await driver.execute_query(
        "CREATE (:Entity {uuid: 'a'})-[:RELATES_TO {uuid: 'x', kept: 1, "
        "cleared: 2, set: 3}]->(:Entity {uuid: 'b'})"
    )
    edge = "MATCH ()-[e:RELATES_TO {uuid: 'x'}]->() "

    await driver.execute_query(
        edge + "SET e += $m", m={"cleared": None, "set": 30, "added": [1, 2]}
    )
    assert (await edge_properties(driver, "x")) == {
        "uuid": "x",
        "kept": 1,
        "set": 30,
        "added": [1, 2],
    }

    await driver.execute_query(edge + "SET e.cleared = 2")
    await driver.execute_query(
        "UNWIND $edges AS edge MATCH ()-[r:RELATES_TO {uuid: edge.uuid}]->() "
        "SET r += edge SET r.fact_embedding = vecf32(edge.fact_embedding)",
        edges=[{"uuid": "x", "cleared": None, "set": 300, "fact_embedding": [0.5]}],
    )
    assert (await edge_properties(driver, "x")) == {
        "uuid": "x",
        "kept": 1,
        "set": 300,
        "added": [1, 2],
    }

    await driver.execute_query(edge + "SET e = $m", m={"uuid": "x", "set": 3000})
    assert (await edge_properties(driver, "x")) == {"uuid": "x", "set": 3000}
