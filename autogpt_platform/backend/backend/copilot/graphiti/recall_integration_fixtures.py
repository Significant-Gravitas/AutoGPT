"""Shared set-up for the live recall-policy tests (``*_integration_test.py``).

Facts go in through graphiti's real ``add_episode`` with only the LLM
boundary scripted (the ``stub_graphiti_client`` fixture), so what the tests
read back is what production would have written. Not collected by pytest.
"""

import asyncio
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from pytest_mock import MockerFixture

from backend.api.features.memory import routes as memory_routes
from backend.copilot.tools import graphiti_search

from . import recall
from .falkordb_driver import AutoGPTFalkorDriver
from .scope import MemoryScope
from .types import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES

# (source entity, target entity, fact sentence)
Fact = tuple[str, str, str]
BuildClient = Callable[[AutoGPTFalkorDriver, dict[str, dict]], Graphiti]

ALICE: Fact = ("Alice", "Atlas", "Alice works on Atlas")
BOB: Fact = ("Bob", "Atlas", "Bob leads Atlas")
CAROL: Fact = ("Carol", "Borealis", "Carol owns Borealis")


class FakeRedis:
    """Honours SET NX like Redis, so ``record_memory_hit`` counts correctly."""

    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    async def get(self, key: str) -> bytes | None:
        return str(self.values[key]).encode() if key in self.values else None

    async def set(self, key: str, value: int, **kwargs: object) -> bool:
        if kwargs.get("nx") and key in self.values:
            return False
        self.values[key] = int(value)
        return True

    async def incr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        return True


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
               e.episodes AS episodes
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
               ep.entity_edges AS entity_edges
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
