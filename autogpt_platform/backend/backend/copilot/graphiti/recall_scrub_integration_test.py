"""What graphiti read out of a forgotten sentence onto entities, through the
production worker against a live FalkorDB.

graphiti keeps a typed attribute an extraction filled from the sentence
(``Person.role``) and a summary on every entity an episode mentions, and
sends both to its attribute, summary and entity resolution prompts. A forget
clears them on every entity the fact joins or its episodes mention
(``recall_hide.scrub_entities``). After it, the only places the sentence
remains are the audit copies: the episode it came from and the edge's
``*_redacted`` fields (plus what the synthetic case seeds into audit-only
metadata). Reproduced first by an independent validation
(``r3-ingestion-independent.py``, ``typed_attribute_scrub`` and
``synthetic_scrub``).

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_scrub_integration_test.py
"""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from .falkordb_driver import AutoGPTFalkorDriver
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BuildClient,
    Fact,
    ingest_through_the_worker,
    live_facts,
    patch_recall_boundaries,
    rows,
    scripted_responses,
    sentence_properties,
    stop_ingestion_workers,
)
from .scope import MemoryScope

# The ``ExtractedEntities`` id of ``Person``: 0 is plain ``Entity``, then
# ``types.ENTITY_TYPES`` in order.
_PERSON = 1
_AUDIT_ONLY = {"Episodic.content", "RELATES_TO.fact_redacted"}


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


def _alice_a_person(facts: list[Fact], *, role: str | None) -> dict[str, dict]:
    """``facts`` extracted with Alice as a ``Person`` whose ``role`` the
    model fills with ``role``."""
    responses = scripted_responses(facts)
    if not facts:
        names = [{"name": "Alice", "entity_type_id": _PERSON}]
        responses["ExtractedEntities"] = {"extracted_entities": names}
    for entity in responses["ExtractedEntities"]["extracted_entities"]:
        if entity["name"] == "Alice":
            entity["entity_type_id"] = _PERSON
    responses["Person"] = {"role": role} if role else {}
    return responses


async def _learn_then_forget(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient
) -> str:
    client = build(driver, _alice_a_person([ALICE], role=ALICE[2]))
    await ingest_through_the_worker(
        driver, scope, client, [ALICE], session_id="s-first"
    )
    [alice] = await live_facts(driver)
    return alice


async def _hello(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient
) -> list[str]:
    """A later episode that only mentions Alice; every message graphiti sent
    its model for it."""
    client = build(driver, _alice_a_person([], role=None))
    answer = client.llm_client._generate_response
    with patch.object(
        client.llm_client, "_generate_response", side_effect=answer
    ) as generate:
        await ingest_through_the_worker(
            driver,
            scope,
            client,
            [],
            session_id="s-hello",
            body="Alice called to say hello.",
        )
    return [m.content for call in generate.await_args_list for m in call.args[0]]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_typed_attribute_holding_the_sentence_is_cleared(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    driver, scope = scope_graph
    alice = await _learn_then_forget(driver, scope, stub_graphiti_client)
    assert "Entity.role" in await sentence_properties(driver, ALICE[2])

    await retract(scope, [alice])

    assert await sentence_properties(driver, ALICE[2]) == _AUDIT_ONLY
    sent = await _hello(driver, scope, stub_graphiti_client)
    assert sent, "graphiti asked its model nothing"
    assert [message for message in sent if ALICE[2] in message] == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_everything_graphiti_keeps_about_the_sentence_is_cleared(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    """Every place graphiti can keep text, seeded with the sentence: entity
    attributes, a serialized ``attributes`` map, a summary on an entity the
    episode only mentions, a community, the edge's name and the episode's
    own metadata. Only audit copies and audit-only metadata keep it."""
    driver, scope = scope_graph
    alice = await _learn_then_forget(driver, scope, stub_graphiti_client)
    await _seed_everywhere(driver, scope, alice)

    await retract(scope, [alice])

    assert await sentence_properties(driver, ALICE[2]) == _AUDIT_ONLY | {
        "Episodic.name",
        "Episodic.source_description",
        "RELATES_TO.name_redacted",
        "Community.name",
    }
    mentioned = await rows(
        driver, "MATCH (n:Entity {name: 'MentionOnly'}) RETURN properties(n) AS p"
    )
    assert mentioned[0]["p"]["summary"] == ""
    sent = await _hello(driver, scope, stub_graphiti_client)
    assert sent and [message for message in sent if ALICE[2] in message] == []


async def _seed_everywhere(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, alice: str
) -> None:
    params: dict[str, Any] = {
        "uuid": alice,
        "sentence": ALICE[2],
        "attributes": json.dumps({"prior_fact": ALICE[2]}),
        "group_id": scope.group_id,
        "community": uuid.uuid4().hex,
        "member": uuid.uuid4().hex,
        "entity": uuid.uuid4().hex,
        "mention": uuid.uuid4().hex,
    }
    for query in _SEEDS:
        await driver.execute_query(query, **params)


# Codex's ``synthetic_scrub`` seeds, one statement each.
_SEEDS = (
    "MATCH (n:Entity) SET n.role = $sentence, n.attributes = $attributes",
    """MATCH (ep:Episodic)
       WHERE $uuid IN ep.entity_edges
       SET ep.name = $sentence, ep.source_description = $sentence""",
    """MATCH (s)-[e:RELATES_TO {uuid: $uuid}]->()
       SET e.name = $sentence
       CREATE (c:Community {uuid: $community, name: $sentence,
                            summary: $sentence, group_id: $group_id})
       CREATE (c)-[:HAS_MEMBER {uuid: $member, group_id: $group_id}]->(s)""",
    """MATCH (ep:Episodic)
       WHERE $uuid IN ep.entity_edges
       CREATE (n:Entity {uuid: $entity, group_id: $group_id,
                         name: 'MentionOnly', summary: $sentence})
       CREATE (ep)-[:MENTIONS {uuid: $mention, group_id: $group_id}]->(n)""",
)
