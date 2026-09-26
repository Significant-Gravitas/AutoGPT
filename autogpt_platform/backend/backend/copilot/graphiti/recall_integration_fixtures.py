"""Shared set-up for the live recall-policy tests (``*_integration_test.py``).

Facts go in through graphiti's real ``add_episode`` with only the LLM
boundary scripted (the ``stub_graphiti_client`` fixture), so what the tests
read back is what production would have written; ``ingest_through_the_worker``
goes one step further out, through the production ingestion worker. Not
collected by pytest.
"""

import asyncio
import json
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from graphiti_core import Graphiti
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.nodes import EpisodeType
from pytest_mock import MockerFixture

from backend.api.features.memory import routes as memory_routes
from backend.copilot.tools import graphiti_search

from . import ingest, recall
from .falkordb_driver import AutoGPTFalkorDriver
from .recall_fake_redis import FakeRedis
from .recall_ingest import ForgetAwareLLMClient
from .scope import MemoryScope
from .types import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES

# (source entity, target entity, fact sentence)
Fact = tuple[str, str, str]
BuildClient = Callable[[AutoGPTFalkorDriver, dict[str, dict]], Graphiti]

ALICE: Fact = ("Alice", "Atlas", "Alice works on Atlas")
BOB: Fact = ("Bob", "Atlas", "Bob leads Atlas")
CAROL: Fact = ("Carol", "Borealis", "Carol owns Borealis")

_INGEST_TIMEOUT_SECONDS = 25.0


def patch_recall_boundaries(
    mocker: MockerFixture, driver: AutoGPTFalkorDriver, build_client: BuildClient
) -> None:
    """Search through a stubbed-LLM Graphiti on the test graph, count hits in
    memory, and report memory as enabled for the settings routes and tools."""
    client = build_client(driver, {})
    mocker.patch.object(
        recall, "get_graphiti_client", mocker.AsyncMock(return_value=client)
    )
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        mocker.AsyncMock(return_value=FakeRedis()),
    )
    enabled = mocker.AsyncMock(return_value=True)
    mocker.patch.object(memory_routes, "is_enabled_for_user", enabled)
    mocker.patch.object(graphiti_search, "is_enabled_for_user", enabled)


def model_of(client: Graphiti) -> LLMClient:
    """The scripted model under ``client``'s forget-aware wrapper
    (``recall_ingest.ForgetAwareLLMClient``), to patch or inspect."""
    return cast(ForgetAwareLLMClient, client.llm_client).inner


def capture_spawned_tasks(mocker: MockerFixture) -> list[asyncio.Task]:
    """Keep the hit-recording tasks ``memory_search`` detaches, so a test can
    await them instead of racing them."""
    spawned: list[asyncio.Task] = []

    def spawn(coro: Coroutine[Any, Any, None], *, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        spawned.append(task)
        return task

    mocker.patch.object(graphiti_search, "spawn_background_task", spawn)
    return spawned


def scripted_responses(facts: list[Fact], *, status: str = "active") -> dict[str, dict]:
    """What the scripted LLM answers so ``add_episode`` extracts ``facts``."""
    names = list(dict.fromkeys(name for s, t, _ in facts for name in (s, t)))
    edges = [
        {
            "source_entity_name": source,
            "target_entity_name": target,
            "relation_type": "MemoryFact",
            "fact": sentence,
            "valid_at": None,
            "invalid_at": None,
        }
        for source, target, sentence in facts
    ]
    return {
        "ExtractedEntities": {
            "extracted_entities": [{"name": n, "entity_type_id": 0} for n in names]
        },
        "NodeResolutions": {
            "entity_resolutions": [
                {"id": i, "name": n, "duplicate_candidate_id": -1}
                for i, n in enumerate(names)
            ]
        },
        "ExtractedEdges": {"edges": edges},
        "MemoryFact": {"status": status, "scope": "real:global"},
        "EdgeDuplicate": {"duplicate_facts": [], "contradicted_facts": []},
        # Asked for an entity with no summary and no new fact in the episode
        # (after a forget blanked it); no answer keeps the summary empty.
        "SummarizedEntities": {"summaries": []},
    }


async def ingest_facts(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    build_client: BuildClient,
    facts: list[Fact],
    *,
    status: str = "active",
    session_id: str | None = None,
    body: str | None = None,
) -> tuple[str, dict[str, str]]:
    """One episode through graphiti's ``add_episode`` holding ``facts``.

    Its text is the fact sentences unless ``body`` is given, and it is named
    like a chat turn of ``session_id`` when one is. Returns the episode uuid
    and each fact sentence's edge uuid.
    """
    client = build_client(driver, scripted_responses(facts, status=status))
    name, description = _episode_labels(facts, session_id)
    result = await client.add_episode(
        name=name,
        episode_body=body or ". ".join(sentence for _, _, sentence in facts),
        source=EpisodeType.text,
        source_description=description,
        reference_time=datetime.now(timezone.utc) - timedelta(minutes=1),
        group_id=scope.group_id,
        entity_types=ENTITY_TYPES,
        edge_types=EDGE_TYPES,
        edge_type_map=EDGE_TYPE_MAP,
    )
    edges = {edge.fact: edge.uuid for edge in result.edges}
    assert set(edges) == {sentence for _, _, sentence in facts}, (
        "add_episode did not persist the scripted facts; re-run with "
        "--log-cli-level=WARNING to see why"
    )
    return result.episode.uuid, edges


async def ingest_through_the_worker(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    client: Graphiti,
    facts: list[Fact],
    *,
    session_id: str,
    body: str | None = None,
) -> str:
    """One chat turn stating ``facts`` (its text ``body`` when given) through
    the production worker (``ingest.enqueue_episode``), with ``client`` as
    the scope's graphiti client. Returns the new episode's uuid.

    The caller cancels the worker afterwards (``stop_ingestion_workers``).
    """
    name, description = _episode_labels(facts, session_id)
    completion = ingest.IngestionCompletion()
    with patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)):
        queued = await ingest.enqueue_episode(
            scope,
            session_id,
            name=name,
            episode_body=body or ". ".join(sentence for _, _, sentence in facts),
            source_description=description,
            completion=completion,
        )
        assert queued, "the ingestion queue refused the episode"
        completion.register()
        done = await ingest.wait_for_ingestion(completion, _INGEST_TIMEOUT_SECONDS)
        assert done, "the ingestion worker did not finish"
    found = await rows(
        driver,
        "MATCH (ep:Episodic {name: $name}) RETURN ep.uuid AS uuid",
        name=name,
    )
    assert len(found) == 1, "the worker did not write the episode"
    return found[0]["uuid"]


async def stop_ingestion_workers() -> None:
    """Cancel the per-scope workers a test's ingestion started."""
    workers = list(ingest._get_loop_state().group_workers.values())
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


def _episode_labels(facts: list[Fact], session_id: str | None) -> tuple[str, str]:
    """``(name, source_description)``, as ingestion writes them for a chat
    turn of ``session_id``."""
    if session_id is None:
        return f"episode-{len(facts)}-{facts[0][2]}", "recall integration test"
    return f"conversation_{session_id}", f"User message in session {session_id}"


async def rows(
    driver: AutoGPTFalkorDriver, query: str, **params: object
) -> list[dict[str, Any]]:
    result = await driver.execute_query(query, **params)
    assert result is not None, "a Cypher read returned no result set"
    return result[0]


async def edge_row(driver: AutoGPTFalkorDriver, edge_uuid: str) -> dict[str, Any]:
    """The edge's audit fields, or ``{}`` once it is gone."""
    found = await rows(
        driver,
        """
        MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
        RETURN e.status AS status, e.expired_at AS expired_at,
               e.invalid_at AS invalid_at, e.expiration_reason AS reason,
               e.forgotten_at AS forgotten_at, e.episodes AS episodes,
               e.fact AS fact, e.fact_redacted AS fact_redacted,
               e.name AS name, e.name_redacted AS name_redacted
        """,
        uuid=edge_uuid,
    )
    return found[0] if found else {}


async def episode_row(driver: AutoGPTFalkorDriver, episode_uuid: str) -> dict[str, Any]:
    """The stored episode, redacted or not, or ``{}`` once it is gone."""
    found = await rows(
        driver,
        """
        MATCH (ep:Episodic {uuid: $uuid})
        RETURN ep.content AS content, ep.redacted_at AS redacted_at,
               ep.entity_edges AS entity_edges, ep.name AS name,
               ep.hard_deleted_at AS hard_deleted_at,
               ep.provenance AS provenance
        """,
        uuid=episode_uuid,
    )
    return found[0] if found else {}


async def count(driver: AutoGPTFalkorDriver, query: str) -> int:
    return int((await rows(driver, query))[0]["c"])


async def recalled_facts(scope: MemoryScope) -> set[str]:
    return {fact.uuid for fact in await recall.search_facts(scope, "Atlas", limit=20)}


async def recalled_episodes(scope: MemoryScope) -> set[str]:
    return {episode.uuid for episode in await recall.recent_episodes(scope, 5)}


async def sentence_properties(driver: AutoGPTFalkorDriver, sentence: str) -> set[str]:
    """Every node and edge property anywhere in the graph that holds
    ``sentence``, as ``Label.property``."""
    nodes = await rows(
        driver, "MATCH (n) RETURN labels(n) AS labels, properties(n) AS properties"
    )
    edges = await rows(
        driver,
        "MATCH ()-[e]->() RETURN [type(e)] AS labels, properties(e) AS properties",
    )
    return {
        f"{_kind(row['labels'])}.{key}"
        for row in nodes + edges
        for key, value in row["properties"].items()
        if sentence in json.dumps(value, default=str)
    }


def _kind(labels: list[str]) -> str:
    known = [label for label in ("Episodic", "Entity", "Community") if label in labels]
    return (known or labels)[0]


async def live_facts(driver: AutoGPTFalkorDriver) -> dict[str, str]:
    """Every live fact's sentence by edge uuid, under the recall policy."""
    found = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO]->() "
        f"WHERE {recall.live_fact_predicate('e')} "
        "RETURN e.uuid AS uuid, e.fact AS fact",
    )
    return {row["uuid"]: row["fact"] for row in found}
