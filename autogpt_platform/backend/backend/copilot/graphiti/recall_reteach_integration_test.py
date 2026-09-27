"""Facts stated again after a forget, several in one episode, through the
production worker against a live FalkorDB.

graphiti's model may call each new statement a duplicate of a different
edge between the same entities, forgotten or live. A forgotten one is never
a duplicate (``recall_ingest.ForgetAwareLLMClient``), so each statement
merged into a forgotten edge is saved as its own live edge, and one merged
into a live edge stays merged, as graphiti decided: no statement is lost or
saved twice, and graphiti extracts the episode once. Reproduced first by an
independent validation (``r3-ingestion-independent.py``, ``multi_merge``;
``r4-ingestion-repair-extra.py``, the mixed forgotten and live pair;
``r5-ingestion-independent.py``, ``corrupt_partial``, a dedup prompt the
guard cannot read in full).

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_reteach_integration_test.py
"""

import ast
import re
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.models import Message
from pydantic import BaseModel
from pytest_mock import MockerFixture

from .falkordb_driver import AutoGPTFalkorDriver
from .recall import FORGOTTEN_FACT
from .recall_forget import retract
from .recall_ingest import ForgetAwareLLMClient
from .recall_integration_fixtures import (
    ALICE,
    BuildClient,
    Fact,
    edge_row,
    episode_row,
    ingest_through_the_worker,
    live_facts,
    model_of,
    patch_recall_boundaries,
    recalled_episodes,
    scripted_responses,
    stop_ingestion_workers,
)
from .scope import MemoryScope

_BUDGET: Fact = ("Alice", "Atlas", "Alice owns the Atlas budget")
_ASSIGNED: Fact = ("Alice", "Atlas", "Alice is assigned to work on the Atlas project")
_MANAGES: Fact = ("Alice", "Atlas", "Alice manages the Atlas budget")
_SECTION = re.compile(r"<EXISTING FACTS>\s*(.*?)\s*</EXISTING FACTS>", re.DOTALL)
_NEW_FACT = re.compile(r"<NEW FACT>\s*(.*?)\s*</NEW FACT>", re.DOTALL)


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


async def _learn(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient, facts
) -> dict[str, str]:
    """``facts`` through the worker: each live fact's sentence by uuid."""
    client = build(driver, _responses(facts, "2026-01-01T00:00:00Z"))
    await ingest_through_the_worker(driver, scope, client, facts, session_id="s-1")
    return await live_facts(driver)


async def _say_again(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    build: BuildClient,
    facts: list[Fact],
) -> tuple[str, list[str]]:
    """``facts`` in a new chat turn, graphiti's model calling a work
    statement a duplicate of the work fact and a budget statement one of the
    budget fact, whether forgotten or live: the new episode's uuid and the
    answer each model call asked for."""
    client = build(driver, _responses(facts, "2026-06-01T00:00:00Z"))
    model = model_of(client)
    answer = model._generate_response

    async def resolve(messages, response_model=None, *args: Any, **kwargs: Any):
        if response_model is not None and response_model.__name__ == "EdgeDuplicate":
            return _merge_by_topic(messages[-1].content)
        return await answer(messages, response_model, *args, **kwargs)

    with patch.object(model, "_generate_response", side_effect=resolve) as calls:
        episode = await ingest_through_the_worker(
            driver, scope, client, facts, session_id="s-2"
        )
    return episode, [call.args[1].__name__ for call in calls.await_args_list]


def _merge_by_topic(prompt: str) -> dict[str, list[int]]:
    """The duplicate a model would name: the existing fact on the new
    statement's topic, the forgotten one read by its audit topic."""
    found = _NEW_FACT.search(prompt)
    budget = found is not None and "budget" in found.group(1)
    section = _SECTION.search(prompt)
    existing = ast.literal_eval(section.group(1)) if section else []
    wanted = _BUDGET[2] if budget else ALICE[2]
    same = [c["idx"] for c in existing if c["fact"] == wanted]
    forgotten = [c["idx"] for c in existing if c["fact"] == FORGOTTEN_FACT]
    return {"duplicate_facts": (same or forgotten)[:1], "contradicted_facts": []}


async def _audits(driver: AutoGPTFalkorDriver, uuids) -> dict[str, dict[str, Any]]:
    return {uuid: await edge_row(driver, uuid) for uuid in uuids}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_forgotten_facts_said_again_both_come_back(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    driver, scope = scope_graph
    forgotten = list(
        await _learn(driver, scope, stub_graphiti_client, [ALICE, _BUDGET])
    )
    await retract(scope, forgotten)
    audits = await _audits(driver, forgotten)

    episode, asked = await _say_again(
        driver, scope, stub_graphiti_client, [_ASSIGNED, _MANAGES]
    )

    live = await live_facts(driver)
    assert sorted(live.values()) == sorted([_ASSIGNED[2], _MANAGES[2]])
    assert await _audits(driver, forgotten) == audits, "a forgotten edge changed"
    assert sorted((await episode_row(driver, episode))["entity_edges"]) == sorted(live)
    assert episode in await recalled_episodes(scope)
    assert asked.count("ExtractedEdges") == 1, "no second extraction"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_statement_merged_into_a_live_fact_is_not_saved_twice(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    """Work is forgotten, the budget fact live: the work statement comes
    back as its own fact, the budget paraphrase stays merged into the live
    budget fact, and no third fact appears."""
    driver, scope = scope_graph
    learned = await _learn(driver, scope, stub_graphiti_client, [ALICE, _BUDGET])
    [work] = [uuid for uuid, fact in learned.items() if fact == ALICE[2]]
    [budget] = [uuid for uuid, fact in learned.items() if fact == _BUDGET[2]]
    await retract(scope, [work])
    audit = await edge_row(driver, work)

    episode, asked = await _say_again(
        driver, scope, stub_graphiti_client, [_ASSIGNED, _MANAGES]
    )

    live = await live_facts(driver)
    assert sorted(live.values()) == sorted([_BUDGET[2], _ASSIGNED[2]])
    assert episode in (await edge_row(driver, budget))["episodes"], "merged"
    assert await edge_row(driver, work) == audit
    assert asked.count("ExtractedEdges") == 1, "no second extraction"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_an_episode_that_only_mentions_the_entity_adds_no_fact(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    driver, scope = scope_graph
    [alice] = await _learn(driver, scope, stub_graphiti_client, [ALICE])
    await retract(scope, [alice])
    audit = await edge_row(driver, alice)
    client = stub_graphiti_client(driver, scripted_responses([]))

    episode = await ingest_through_the_worker(
        driver, scope, client, [], session_id="s-2", body="Alice called to say hello."
    )

    assert await live_facts(driver) == {}
    assert await edge_row(driver, alice) == audit
    assert episode in await recalled_episodes(scope)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_dedup_prompt_the_guard_cannot_read_saves_the_statement_as_new(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    """The existing-facts tags renamed on the way to the guard, the model
    naming the forgotten edge a duplicate: the statement is saved as its own
    live fact and the forgotten edge keeps its marker and audit copies."""
    driver, scope = scope_graph
    [alice] = await _learn(driver, scope, stub_graphiti_client, [ALICE])
    await retract(scope, [alice])
    audit = await edge_row(driver, alice)
    responses = _responses([_ASSIGNED], "2026-06-01T00:00:00Z")
    responses["EdgeDuplicate"] = {"duplicate_facts": [0], "contradicted_facts": []}
    client = stub_graphiti_client(driver, responses)
    guard = ForgetAwareLLMClient.generate_response

    async def renamed(
        self: ForgetAwareLLMClient,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if response_model is EdgeDuplicate:
            messages = [
                m.model_copy(
                    update={"content": m.content.replace("EXISTING FACTS", "RENAMED")}
                )
                for m in messages
            ]
        return await guard(self, messages, response_model, *args, **kwargs)

    with patch.object(ForgetAwareLLMClient, "generate_response", renamed):
        episode = await ingest_through_the_worker(
            driver, scope, client, [_ASSIGNED], session_id="s-2"
        )

    assert list((await live_facts(driver)).values()) == [_ASSIGNED[2]]
    assert await edge_row(driver, alice) == audit, "the forgotten edge changed"
    assert episode in await recalled_episodes(scope)
