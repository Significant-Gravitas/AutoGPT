"""Ingestion under the recall policy, against a live FalkorDB: what graphiti's
own prompts are shown after a forget.

graphiti's ``add_episode`` shows its extraction prompts the newest earlier
episodes as context. Left to pick them itself it cannot see a forget, so a
forgotten episode's text went back to the extractor with every later one.
Its entity resolution also reads the summaries of the entities a new episode
may be about, which graphiti wrote from fact sentences, and its edge
resolution offers every existing edge, the retracted one included, as a
duplicate or contradiction candidate. A forget therefore leaves a placeholder
where the sentence was and blanks those summaries (``recall_hide.py``).

These drive the production worker (``ingest.enqueue_episode``) with only the
LLM boundary scripted and read every message the model was sent.
``recall_ingest_integration_test.py`` covers a forgotten fact stated again.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/ingest_recall_integration_test.py
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from . import ingest
from .falkordb_driver import AutoGPTFalkorDriver
from .recall import FORGOTTEN_FACT
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BOB,
    CAROL,
    ingest_facts,
    ingest_through_the_worker,
    live_facts,
    rows,
    scripted_responses,
    stop_ingestion_workers,
)
from .scope import MemoryScope

_INGEST_TIMEOUT_SECONDS = 25.0
# Text only an episode body holds: no fact, entity or summary repeats it, so
# a prompt can only have it from the episode itself.
_FORGOTTEN_NOTE = "She said so over lunch on the twelfth."
_KEPT_NOTE = "Carol mentioned it during the Monday stand-up."


@pytest_asyncio.fixture(loop_scope="function")
async def ingest_worker_cleanup(mocker: MockerFixture) -> AsyncIterator[None]:
    """No dream registration on a first write; no idle worker left behind."""
    mocker.patch(
        "backend.copilot.dream.scheduling.ensure_dream_system_scheduled",
        AsyncMock(return_value=None),
    )
    yield
    await stop_ingestion_workers()


async def _ingest_through_the_worker(scope: MemoryScope, body: str) -> None:
    completion = ingest.IngestionCompletion()
    queued = await ingest.enqueue_episode(
        scope, "session-bob", name="bob", episode_body=body, completion=completion
    )
    assert queued
    completion.register()
    assert await ingest.wait_for_ingestion(completion, _INGEST_TIMEOUT_SECONDS)


async def _summaries(driver: AutoGPTFalkorDriver) -> dict[str, str]:
    found = await rows(
        driver, "MATCH (n:Entity) RETURN n.name AS name, n.summary AS summary"
    )
    return {row["name"]: row["summary"] or "" for row in found}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extraction_is_never_shown_a_forgotten_episode(
    scope_graph, stub_graphiti_client, mocker, ingest_worker_cleanup
) -> None:
    driver, scope = scope_graph
    forgotten, edges = await ingest_facts(
        driver,
        scope,
        stub_graphiti_client,
        [ALICE],
        body=f"{ALICE[2]}. {_FORGOTTEN_NOTE}",
    )
    kept, _ = await ingest_facts(
        driver, scope, stub_graphiti_client, [CAROL], body=f"{CAROL[2]}. {_KEPT_NOTE}"
    )
    await retract(scope, [edges[ALICE[2]]])
    client = stub_graphiti_client(driver, scripted_responses([BOB]))
    generate = mocker.spy(client.llm_client, "_generate_response")
    add_episode = mocker.spy(client, "add_episode")
    mocker.patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client))

    await _ingest_through_the_worker(scope, BOB[2])

    prompts = [m.content for call in generate.call_args_list for m in call.args[0]]
    assert any(_KEPT_NOTE in prompt for prompt in prompts), "no earlier episodes"
    leaked = [prompt for prompt in prompts if _FORGOTTEN_NOTE in prompt]
    assert leaked == [], "the extractor was shown a forgotten episode"
    context = add_episode.call_args.kwargs["previous_episode_uuids"]
    assert kept in context and forgotten not in context


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_later_prompt_carries_a_forgotten_sentence(
    scope_graph, stub_graphiti_client, mocker, ingest_worker_cleanup
) -> None:
    """Ingest Alice, forget her, ingest Bob: not one message graphiti sends
    the model for Bob holds the Alice sentence, although its entity and edge
    resolution both ran over what the forget left behind."""
    driver, scope = scope_graph
    first = stub_graphiti_client(driver, scripted_responses([ALICE]))
    await ingest_through_the_worker(driver, scope, first, [ALICE], session_id="s-alice")
    [alice] = await live_facts(driver)
    assert any(ALICE[2] in summary for summary in (await _summaries(driver)).values())

    await retract(scope, [alice])

    assert set((await _summaries(driver)).values()) == {""}, "summaries blanked"
    client = stub_graphiti_client(driver, scripted_responses([BOB]))
    generate = mocker.spy(client.llm_client, "_generate_response")
    await ingest_through_the_worker(driver, scope, client, [BOB], session_id="s-bob")

    calls = generate.call_args_list
    asked = {call.args[1].__name__ for call in calls if call.args[1] is not None}
    assert {"NodeResolutions", "EdgeDuplicate"} <= asked, "both resolutions ran"
    prompts = [m.content for call in calls for m in call.args[0]]
    assert any(FORGOTTEN_FACT in prompt for prompt in prompts), "edge offered"
    assert [prompt for prompt in prompts if ALICE[2] in prompt] == []
    assert list((await live_facts(driver)).values()) == [BOB[2]]
