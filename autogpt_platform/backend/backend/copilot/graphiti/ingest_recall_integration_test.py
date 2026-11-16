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
LLM boundary scripted and read every message the model was sent. A status of
``retracted`` the extraction model gives a fact is no forget: the worker puts
it back to ``active``. Nor is a forget's marker it invents: none is saved.
``recall_ingest_integration_test.py`` covers a forgotten fact stated again.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/ingest_recall_integration_test.py
"""

from collections.abc import AsyncIterator
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from graphiti_core.edges import EntityEdge
from pytest_mock import MockerFixture

from backend.copilot.model import ChatSession
from backend.copilot.tools import graphiti_forget
from backend.copilot.tools.models import (
    MemoryForgetCandidatesResponse,
    MemoryForgetConfirmResponse,
)

from . import ingest
from .falkordb_driver import AutoGPTFalkorDriver
from .migrations.backfill_legacy_forgets import ForgetMarkers, scan_markers
from .recall import FORGOTTEN_FACT, is_forgotten, is_live
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BOB,
    CAROL,
    edge_row,
    episode_row,
    ingest_facts,
    ingest_through_the_worker,
    live_facts,
    model_of,
    patch_recall_boundaries,
    rows,
    scripted_responses,
    stop_ingestion_workers,
)
from .scope import MemoryScope
from .types import POLICY_EDGE_FIELDS

_INGEST_TIMEOUT_SECONDS = 25.0
# Text only an episode body holds: no fact, entity or summary repeats it, so
# a prompt can only have it from the episode itself.
_FORGOTTEN_NOTE = "She said so over lunch on the twelfth."
_KEPT_NOTE = "Carol mentioned it during the Monday stand-up."


@pytest_asyncio.fixture(loop_scope="function")
async def ingest_worker_cleanup(mocker: MockerFixture) -> AsyncIterator[None]:
    """No dream registration on a first write; no idle worker left behind."""
    mocker.patch(
        "backend.copilot.graphiti.ingest.ensure_scope_scheduled",
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
    generate = mocker.spy(model_of(client), "_generate_response")
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
    generate = mocker.spy(model_of(client), "_generate_response")
    await ingest_through_the_worker(driver, scope, client, [BOB], session_id="s-bob")

    calls = generate.call_args_list
    asked = {call.args[1].__name__ for call in calls if call.args[1] is not None}
    assert {"NodeResolutions", "EdgeDuplicate"} <= asked, "both resolutions ran"
    prompts = [m.content for call in calls for m in call.args[0]]
    assert any(FORGOTTEN_FACT in prompt for prompt in prompts), "edge offered"
    assert [prompt for prompt in prompts if ALICE[2] in prompt] == []
    assert list((await live_facts(driver)).values()) == [BOB[2]]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_retraction_the_extraction_gives_is_no_forget(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    """ "I retracted my application at Acme": graphiti does not hold its
    extraction model to ``MemoryFact``'s statuses, and this one answers
    ``retracted``. Only a forget retracts, so the worker puts the fact back
    to active: it stays live, and its episode is not redacted."""
    driver, scope = scope_graph
    client = stub_graphiti_client(
        driver, scripted_responses([ALICE], status="retracted")
    )

    episode = await ingest_through_the_worker(
        driver, scope, client, [ALICE], session_id="s-alice"
    )

    [alice] = await live_facts(driver)
    row = await edge_row(driver, alice)
    assert (row["status"], row["forgotten_at"], row["fact"]) == (
        "active",
        None,
        ALICE[2],
    )
    assert (await episode_row(driver, episode))["redacted_at"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_marker_the_extraction_invents_is_never_saved(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, mocker: MockerFixture
) -> None:
    """Codex's probe: graphiti-core 0.30.2 saves every key its extraction
    model answers, and this one answers a ``forgotten_at`` and audit copies
    holding a sentence nobody stated. The worker saves none of them: the fact
    stays live, and nothing ``memory_forget_search`` or
    ``memory_forget_confirm`` answers holds the sentence, before the fact is
    forgotten or after."""
    invented = "Alice works on the secret Atlas plan"
    driver, scope = scope_graph
    responses = scripted_responses([BOB])
    responses["MemoryFact"] |= dict.fromkeys(POLICY_EDGE_FIELDS, invented)
    client = stub_graphiti_client(driver, responses)

    await ingest_through_the_worker(driver, scope, client, [BOB], session_id="s-bob")

    [bob] = await live_facts(driver)
    stored = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() RETURN properties(e) AS p",
        uuid=bob,
    )
    assert POLICY_EDGE_FIELDS.isdisjoint(stored[0]["p"])
    [edge] = await EntityEdge.get_by_uuids(driver, [bob])
    assert is_live(edge) and not is_forgotten(edge)

    patch_recall_boundaries(mocker, driver, stub_graphiti_client)
    user, session = scope.owner_user_id, ChatSession.new(
        scope.owner_user_id, dry_run=False
    )
    search = graphiti_forget.MemoryForgetSearchTool()
    confirm = graphiti_forget.MemoryForgetConfirmTool()
    live = await search._execute(user, session, query="Atlas", include_forgotten=True)
    soft = await confirm._execute(user, session, uuids=[bob])
    gone = await search._execute(user, session, query="Atlas", include_forgotten=True)
    hard = await confirm._execute(user, session, uuids=[bob], hard_delete=True)

    assert isinstance(live, MemoryForgetCandidatesResponse)
    assert [(c["uuid"], c.get("forgotten")) for c in live.candidates] == [(bob, None)]
    assert isinstance(gone, MemoryForgetCandidatesResponse)
    [candidate] = gone.candidates
    assert (candidate["uuid"], candidate["forgotten"]) == (bob, "true")
    assert datetime.fromisoformat(candidate["forgotten_at"]).tzinfo is not None
    for done in (soft, hard):
        assert isinstance(done, MemoryForgetConfirmResponse)
        assert done.deleted_uuids == [bob]
    for answer in (live, soft, gone, hard):
        assert invented not in answer.model_dump_json()
    assert await edge_row(driver, bob) == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_marker_scan_tells_an_invented_marker_from_a_forget(
    scope_graph, stub_graphiti_client
) -> None:
    """The pre-deploy scan (``backfill_legacy_forgets --scan-markers``) lists
    both marked facts by uuid, and as unscrubbed only the one whose marker a
    past extraction invented: a forget scrubs the sentence it marks."""
    driver, scope = scope_graph
    _, edges = await ingest_facts(driver, scope, stub_graphiti_client, [ALICE, BOB])
    alice, bob = edges[ALICE[2]], edges[BOB[2]]
    await retract(scope, [alice])
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e.forgotten_at = $invented",
        uuid=bob,
        invented="2026-01-01T00:00:00+00:00",
    )

    found = await scan_markers(scope.group_id)

    assert found == [
        ForgetMarkers(
            graph=scope.group_id, marked=sorted([alice, bob]), unscrubbed=[bob]
        )
    ]
