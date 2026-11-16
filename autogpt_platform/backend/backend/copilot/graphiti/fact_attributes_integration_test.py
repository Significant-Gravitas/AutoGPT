"""A statement merged into a stored fact leaves every attribute on it as it
was (``fact_attributes.py``), through the production worker against a live
FalkorDB. That covers the platform's stamps, the dream's envelope metadata
and the status, whatever graphiti's model answers, on both of graphiti's
merge branches and its exact-text path. A forgotten fact is still no merge
target, a save of a copy read before the forget changes only its episode
list, and FalkorDB runs both saves as the driver rewrites them.

Reproduced first by a probe of the same shape before the fix: every stamp
was lost on a merge, and the status was the model's answer. The stale save
was reproduced first by an independent review of the merge: it put the
forgotten sentence back under the marker, and a later paraphrase went into
the forgotten fact.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/fact_attributes_integration_test.py
"""

import ast
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_entity_edge_save_query,
)
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
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
    rows,
    scripted_responses,
    stop_ingestion_workers,
)
from .scope import MemoryScope
from .types import MemoryFact

_PARAPHRASE: Fact = ("Alice", "Atlas", "Alice is working on Atlas")
_TENTATIVE = {"status": "tentative", "scope": "real:global"}

# What the platform writes on a live fact outside graphiti's extraction: the
# recall stamps, the ratification and web-verification stamps, the dream's
# envelope metadata (``ingest._stamp_edge_metadata``) and its derivation
# record (``recall_derivation.record``).
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
    "derived_from_facts": ["fact-cited"],
    "derived_from_episodes": ["episode-cited"],
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
    duplicate_of: set[str],
) -> tuple[str, list[str]]:
    """``fact`` in a new turn, graphiti's model naming the first candidate
    whose text is in ``duplicate_of`` a duplicate and answering
    ``_TENTATIVE`` for its attributes: the new episode and the models
    asked."""
    responses = _responses(fact, "MemoryFact", "2026-06-01T00:00:00Z")
    responses["MemoryFact"] = dict(_TENTATIVE)
    client = build(driver, responses)
    model = model_of(client)
    scripted = model._generate_response

    async def resolve(messages, response_model=None, *args: Any, **kwargs: Any):
        if response_model is not None and response_model.__name__ == "EdgeDuplicate":
            listed = _existing_facts(messages[-1].content)
            same = [c["idx"] for c in listed if c["fact"] in duplicate_of]
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
        driver, scope, stub_graphiti_client, _PARAPHRASE, duplicate_of={ALICE[2]}
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
async def test_an_attribute_less_stored_memory_fact_seeds_attributes_on_restatement(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    driver, scope = scope_graph
    uuid = await _stamped_fact(driver, scope, stub_graphiti_client, "MemoryFact")
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e += $cleared",
        uuid=uuid,
        cleared={key: None for key in set(STAMPS) | set(MemoryFact.model_fields)},
    )
    before = await edge_properties(driver, uuid)
    assert not (set(before) & (set(STAMPS) | set(MemoryFact.model_fields)))

    episode, asked = await _restate(
        driver, scope, stub_graphiti_client, _PARAPHRASE, duplicate_of={ALICE[2]}
    )

    after = await edge_properties(driver, uuid)
    assert "MemoryFact" in asked
    assert after["status"] == "tentative"
    assert after["scope"] == "real:global"
    assert episode in after["episodes"]
    seeded = set(MemoryFact.model_fields) | {"episodes"}
    assert {k: v for k, v in after.items() if k not in seeded} == {
        k: v for k, v in before.items() if k not in seeded
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_same_sentence_again_leaves_every_attribute_as_it_was(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    """graphiti's exact-text match, which runs before its model."""
    driver, scope = scope_graph
    uuid = await _stamped_fact(driver, scope, stub_graphiti_client, "MemoryFact")

    episode, asked = await _restate(
        driver, scope, stub_graphiti_client, ALICE, duplicate_of={ALICE[2]}
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
        driver,
        scope,
        stub_graphiti_client,
        _PARAPHRASE,
        duplicate_of={FORGOTTEN_FACT},
    )

    assert await edge_properties(driver, uuid) == before
    assert sorted((await live_facts(driver)).values()) == [_PARAPHRASE[2]]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["single", "bulk"])
async def test_a_stale_save_after_a_forget_links_its_episode_and_nothing_else(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, form: str
) -> None:
    """With no write lock (Redis unreachable, or a lease lost) an ingestion
    saves the fact as it read it before a forget, with a new episode and an
    embedding of its own, by itself or with its entities as ``add_episode``
    does: the fact keeps the forget's scrubbed text, status, dates, audit
    copies and embedding, and only its episode list changes. A later
    paraphrase, which graphiti's model calls a duplicate of it, is then a
    new live fact, not hidden under the forget."""
    driver, scope = scope_graph
    uuid = await _stamped_fact(driver, scope, stub_graphiti_client, "MemoryFact")
    stale = await EntityEdge.get_by_uuid(driver, uuid)
    entities = await EntityNode.get_by_uuids(
        driver, [stale.source_node_uuid, stale.target_node_uuid]
    )
    await retract(scope, [uuid])
    forgotten = await edge_properties(driver, uuid)
    embedding = await _embedding(driver, uuid)
    stale.episodes = [*stale.episodes, "stale-episode"]
    stale.fact_embedding = [-value for value in embedding]

    await _save_stale(driver, stub_graphiti_client, stale, entities, form)

    assert forgotten["fact"] == FORGOTTEN_FACT and stale.fact == ALICE[2]
    assert await edge_properties(driver, uuid) == {
        **forgotten,
        "episodes": stale.episodes,
    }
    assert await _embedding(driver, uuid) == embedding
    await _restate(
        driver,
        scope,
        stub_graphiti_client,
        _PARAPHRASE,
        duplicate_of={ALICE[2], FORGOTTEN_FACT},
    )
    assert sorted((await live_facts(driver)).values()) == [_PARAPHRASE[2]]
    assert await edge_properties(driver, uuid) == {
        **forgotten,
        "episodes": stale.episodes,
    }


async def _save_stale(
    driver: AutoGPTFalkorDriver,
    build: BuildClient,
    stale: EntityEdge,
    entities: list[EntityNode],
    form: str,
) -> None:
    """Save ``stale`` as graphiti does: by itself (``EntityEdge.save``), or
    with its entities in ``add_episode``'s bulk save, through the driver's
    session."""
    if form == "single":
        await stale.save(driver)
        return
    embedder = build(driver, {}).embedder
    await add_nodes_and_edges_bulk(driver, [], [], entities, [stale], embedder)


async def _embedding(driver: AutoGPTFalkorDriver, uuid: str) -> list[float]:
    [row] = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() RETURN e.fact_embedding AS v",
        uuid=uuid,
    )
    return list(row["v"])


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["single", "bulk"])
async def test_falkordb_runs_both_saves_as_the_driver_rewrites_them(
    scope_graph, form: str
) -> None:
    """graphiti's save of three facts as the driver runs it, by itself
    (``execute_query``) or in bulk (``session()``). On a live fact a
    property graphiti's map does not name stays, a null in the map removes
    one, and the rest and the embedding are set; on a fact a forget marked
    only ``episodes`` is; and a fact not stored yet is created from the
    map."""
    driver, _ = scope_graph
    await driver.execute_query(
        "CREATE (a:Entity {uuid: 'a'})-[:RELATES_TO {uuid: 'live', kept: 1, "
        "cleared: 2, set: 3, episodes: ['e1']}]->(b:Entity {uuid: 'b'}) "
        "CREATE (a)-[:RELATES_TO {uuid: 'marked', forgotten_at: 't', "
        "fact: $placeholder, kept: 1, cleared: 2, set: 3, episodes: ['e1']}]->(b)",
        placeholder=FORGOTTEN_FACT,
    )
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() SET e.fact_embedding = vecf32([0.25])"
    )
    fields = {
        "fact": ALICE[2],
        "cleared": None,
        "set": 30,
        "episodes": ["e1", "e2"],
        "fact_embedding": [0.5],
    }

    ends = await _graphiti_saves(driver, form, ["live", "marked", "new"], fields)

    saved = {"fact": ALICE[2], "set": 30, "episodes": ["e1", "e2"], **ends}
    assert await edge_properties(driver, "live") == {"uuid": "live", "kept": 1, **saved}
    assert await edge_properties(driver, "marked") == {
        "uuid": "marked",
        "forgotten_at": "t",
        "fact": FORGOTTEN_FACT,
        "kept": 1,
        "cleared": 2,
        "set": 3,
        "episodes": ["e1", "e2"],
    }
    assert await edge_properties(driver, "new") == {"uuid": "new", **saved}
    assert [await _embedding(driver, u) for u in ("live", "marked", "new")] == [
        [0.5],
        [0.25],
        [0.5],
    ]


async def _graphiti_saves(
    driver: AutoGPTFalkorDriver, form: str, uuids: list[str], fields: dict[str, Any]
) -> dict[str, str]:
    """Save ``fields`` on a fact from ``a`` to ``b`` per uuid with graphiti's
    own save query, as graphiti runs it; the endpoint keys its map holds."""
    falkor = GraphProvider.FALKORDB
    if form == "single":
        ends = {"source_uuid": "a", "target_uuid": "b"}
        for uuid in uuids:
            await driver.execute_query(
                get_entity_edge_save_query(falkor),
                edge_data={**ends, "uuid": uuid, **fields},
            )
        return ends
    ends = {"source_node_uuid": "a", "target_node_uuid": "b"}
    session = driver.session()
    try:
        await session.run(
            get_entity_edge_save_bulk_query(falkor),
            entity_edges=[{**ends, "uuid": uuid, **fields} for uuid in uuids],
        )
    finally:
        await session.close()
    return ends
