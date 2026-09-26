"""Ingestion under the recall policy, against a live FalkorDB.

graphiti's ``add_episode`` shows its extraction prompts the newest earlier
episodes as context. Left to pick them itself it cannot see a forget, so a
forgotten episode's text went back to the extractor with every later one.
This drives the production worker (``ingest.enqueue_episode``) with only the
LLM boundary scripted and reads what the extractor was shown.

Two other channels are out of this policy's reach and stay open: graphiti's
entity resolution shows entity summaries, which a forget does not scrub, and
its edge resolution offers every existing edge, the retracted one included,
as a contradiction candidate. So the forgotten fact's own sentence can still
reach those two prompts; the test pins the episode text.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/ingest_recall_integration_test.py
"""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from . import ingest
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BOB,
    CAROL,
    ingest_facts,
    scripted_responses,
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
    workers = list(ingest._get_loop_state().group_workers.values())
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


async def _ingest_through_the_worker(scope: MemoryScope, body: str) -> None:
    completion = ingest.IngestionCompletion()
    queued = await ingest.enqueue_episode(
        scope, "session-bob", name="bob", episode_body=body, completion=completion
    )
    assert queued
    completion.register()
    assert await ingest.wait_for_ingestion(completion, _INGEST_TIMEOUT_SECONDS)


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
