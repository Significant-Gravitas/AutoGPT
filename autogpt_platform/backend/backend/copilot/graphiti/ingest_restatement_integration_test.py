"""A user's chat turn promotes the tentative facts it states, and nothing
else does (``ingest._promote_restated``), through the production worker
against a live FalkorDB. The turn promotes a never-recalled dream proposal
it restates and a fact of its own that graphiti's model labelled tentative.
It leaves alone the fact it invalidates, and an assistant's finding, a
memory-store call or a dream write promotes nothing.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/ingest_restatement_integration_test.py
"""

import ast
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from . import ingest
from .falkordb_driver import AutoGPTFalkorDriver
from .memory_model import MemoryEnvelope, MemoryKind, MemoryStatus, SourceKind
from .recall_citations import Citations
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

_BUDGET: Fact = ("Alice", "Atlas", "Alice owns the Atlas budget")
_RESTATED: Fact = ("Alice", "Atlas", "Alice is working on Atlas")
_NEW: Fact = ("Alice", "Atlas", "Alice leads the Atlas launch")
_CONTRADICTS: Fact = ("Alice", "Atlas", "Alice handed the Atlas budget to Bob")
# Long enough for the finding lane (``ingest._MIN_FINDING_LENGTH``).
_FINDING = (
    "Based on the project notes, Alice is working on Atlas and leads the Atlas "
    "launch this quarter, with the budget review for the whole programme "
    "scheduled for next month."
)
_PROPOSAL = {
    "status": "tentative",
    "source_kind": "assistant_derived",
    "provenance": "dream:pass-1",
}
_WAIT_SECONDS = 25.0


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)
    mocker.patch.object(ingest, "resolve_user_name", AsyncMock(return_value="Alice"))


@pytest_asyncio.fixture(loop_scope="function")
async def ingest_worker_cleanup(mocker: MockerFixture) -> AsyncIterator[None]:
    mocker.patch(
        "backend.copilot.dream.registry.ensure_dream_system_scheduled",
        AsyncMock(return_value=None),
    )
    yield
    await stop_ingestion_workers()


async def _proposals(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient
) -> dict[str, str]:
    """``ALICE`` and ``_BUDGET`` learnt, then made never-recalled dream
    proposals: each one's uuid by its sentence."""
    responses = scripted_responses([ALICE, _BUDGET])
    for edge in responses["ExtractedEdges"]["edges"]:
        edge["valid_at"] = "2026-01-01T00:00:00Z"
    client = build(driver, responses)
    await ingest_through_the_worker(
        driver, scope, client, [ALICE, _BUDGET], session_id="s-1"
    )
    learnt = {fact: uuid for uuid, fact in (await live_facts(driver)).items()}
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN $uuids SET e += $proposal",
        uuids=list(learnt.values()),
        proposal=_PROPOSAL,
    )
    return learnt


def _turn_client(
    mocker: MockerFixture,
    driver: AutoGPTFalkorDriver,
    build: BuildClient,
    facts: list[Fact],
):
    """A client extracting ``facts``, its model labelling each new one
    tentative, calling ``_RESTATED`` a duplicate of ``ALICE`` and
    ``_CONTRADICTS`` a contradiction of ``_BUDGET``."""
    responses = scripted_responses(facts, status="tentative")
    for edge in responses["ExtractedEdges"]["edges"]:
        edge["valid_at"] = "2026-06-01T00:00:00Z"
    client = build(driver, responses)
    model = model_of(client)
    scripted = model._generate_response

    async def resolve(messages, response_model=None, *args: Any, **kwargs: Any):
        if response_model is not None and response_model.__name__ == "EdgeDuplicate":
            return _decide(messages[-1].content)
        return await scripted(messages, response_model, *args, **kwargs)

    mocker.patch.object(model, "_generate_response", side_effect=resolve)
    return client


def _decide(prompt: str) -> dict[str, list[int]]:
    new = _between(prompt, "NEW FACT")
    existing = ast.literal_eval(_between(prompt, "EXISTING FACTS"))
    idx = {candidate["fact"]: candidate["idx"] for candidate in existing}
    if new == _RESTATED[2] and ALICE[2] in idx:
        return {"duplicate_facts": [idx[ALICE[2]]], "contradicted_facts": []}
    if new == _CONTRADICTS[2] and _BUDGET[2] in idx:
        return {"duplicate_facts": [], "contradicted_facts": [idx[_BUDGET[2]]]}
    return {"duplicate_facts": [], "contradicted_facts": []}


def _between(prompt: str, tag: str) -> str:
    start = prompt.index(f"<{tag}>") + len(tag) + 2
    return prompt[start : prompt.index(f"</{tag}>")].strip()


async def _chat_turn(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    client,
    *,
    lane: str,
    user_msg: str,
    assistant_msg: str = "",
) -> str:
    """One chat turn through ``enqueue_conversation_turn`` and the worker,
    only its ``lane`` episode (``conversation_`` for the user's words,
    ``finding_`` for the assistant's finding) written: that episode's uuid."""
    completion = ingest.IngestionCompletion()
    enqueue = ingest._enqueue_payload
    queued: list[str] = []

    async def one_lane(target: MemoryScope, payload: dict) -> bool:
        if not payload["name"].startswith(lane):
            return True
        payload["_completion"] = completion
        queued.append(payload["name"])
        return await enqueue(target, payload)

    with (
        patch.object(ingest, "_enqueue_payload", side_effect=one_lane),
        patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)),
    ):
        await ingest.enqueue_conversation_turn(
            scope.owner_user_id, "s-2", user_msg, assistant_msg
        )
        assert queued == [f"{lane}s-2"], "the turn made no such episode"
        completion.register()
        assert await ingest.wait_for_ingestion(completion, _WAIT_SECONDS)
    [episode] = await rows(
        driver,
        "MATCH (ep:Episodic {name: $name}) RETURN ep.uuid AS uuid",
        name=f"{lane}s-2",
    )
    return episode["uuid"]


async def _fact(driver: AutoGPTFalkorDriver, sentence: str) -> dict[str, Any]:
    [found] = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO {fact: $fact}]->() RETURN e.uuid AS uuid",
        fact=sentence,
    )
    return await edge_properties(driver, found["uuid"])


def _still_a_proposal(props: dict[str, Any]) -> bool:
    return props["status"] == "tentative" and "ratified_at" not in props


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_users_turn_promotes_what_it_states_and_nothing_else(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, mocker: MockerFixture
) -> None:
    """The proposal the user restates and their own new fact become active
    with ``ratified_at``; the proposal their turn contradicts stays as
    graphiti left it, tentative and expired."""
    driver, scope = scope_graph
    learnt = await _proposals(driver, scope, stub_graphiti_client)
    facts = [_RESTATED, _NEW, _CONTRADICTS]
    client = _turn_client(mocker, driver, stub_graphiti_client, facts)

    episode = await _chat_turn(
        driver,
        scope,
        client,
        lane="conversation_",
        user_msg=". ".join(sentence for _, _, sentence in facts),
    )

    restated = await edge_properties(driver, learnt[ALICE[2]])
    assert episode in restated["episodes"], "merged into the proposal"
    assert restated["status"] == "active" and restated["ratified_at"]
    assert restated["source_kind"] == "assistant_derived", "only the status moved"
    assert "recall_count" not in restated, "saying a fact is no recall of it"
    for sentence in (_NEW[2], _CONTRADICTS[2]):
        own = await _fact(driver, sentence)
        assert own["status"] == "active" and own["ratified_at"], sentence
    contradicted = await edge_properties(driver, learnt[_BUDGET[2]])
    assert episode not in contradicted["episodes"]
    assert contradicted["expired_at"], "graphiti invalidated it"
    assert _still_a_proposal(contradicted)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_an_assistants_finding_promotes_nothing(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, mocker: MockerFixture
) -> None:
    """The finding lane's episode, as ``enqueue_conversation_turn`` builds it:
    neither the proposal it restates nor its own tentative fact moves."""
    driver, scope = scope_graph
    learnt = await _proposals(driver, scope, stub_graphiti_client)
    client = _turn_client(mocker, driver, stub_graphiti_client, [_RESTATED, _NEW])

    episode = await _chat_turn(
        driver,
        scope,
        client,
        lane="finding_",
        user_msg="What is Alice doing?",
        assistant_msg=_FINDING,
    )

    restated = await edge_properties(driver, learnt[ALICE[2]])
    assert episode in restated["episodes"]
    assert _still_a_proposal(restated)
    assert _still_a_proposal(await _fact(driver, _NEW[2]))


@pytest.mark.integration
@pytest.mark.parametrize("writer", ["memory_store", "dream"])
@pytest.mark.asyncio
async def test_a_memory_store_call_or_a_dream_write_promotes_nothing(
    scope_graph,
    stub_graphiti_client,
    ingest_worker_cleanup,
    mocker: MockerFixture,
    writer: str,
) -> None:
    driver, scope = scope_graph
    learnt = await _proposals(driver, scope, stub_graphiti_client)
    client = _turn_client(mocker, driver, stub_graphiti_client, [_RESTATED])
    envelope = MemoryEnvelope(
        user="Alice",
        content=_RESTATED[2],
        source_kind=SourceKind.user_asserted,
        memory_kind=MemoryKind.fact,
        status=MemoryStatus.active,
    )
    dream = writer == "dream"
    completion = ingest.IngestionCompletion()

    with patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)):
        assert await ingest.enqueue_episode(
            scope,
            "s-2",
            name="dream_pass-1_consolidate_001" if dream else "memory",
            episode_body=envelope.model_dump_json(),
            is_json=True,
            edge_metadata=dict(_PROPOSAL, scope="real:global") if dream else None,
            citations=Citations(fact_uuids=[learnt[ALICE[2]]]) if dream else None,
            completion=completion,
        )
        completion.register()
        assert await ingest.wait_for_ingestion(completion, _WAIT_SECONDS)

    restated = await edge_properties(driver, learnt[ALICE[2]])
    assert len(restated["episodes"]) == 2, "merged into the proposal"
    assert _still_a_proposal(restated)
