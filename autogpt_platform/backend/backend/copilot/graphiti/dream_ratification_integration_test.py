"""End-to-end dream test: envelope → tentative edge → ratification (SECRT-2482).

Every existing ratification suite pre-fabricates its ``status='tentative'``
rows — ``dream/ratification_test.py`` mocks the FalkorDB driver outright, and
the edge-stamping tests hand-build edges. Nothing asserted that a real
``MemoryEnvelope`` pushed through the real ingestion path actually LANDS a
tentative edge carrying the envelope's ``source_kind``/``provenance``, which
is how the edge-attribute no-op bug fixed in #13390 went undetected.

This file closes that gap. It drives the production write path
(``dream/apply._write_proposed_finding`` → ``ingest.enqueue_episode`` →
``Graphiti.add_episode`` → ``ingest._stamp_edge_metadata``) against a live
FalkorDB, then runs both ratification legs on the edge that path produced:
``try_ratify_on_hit`` (promote on a warm-context hit) and
``run_ratification_pass`` (the nightly sweep that supersedes a tentative edge
which never earned a hit before its grace period elapsed).

Only the LLM boundary is stubbed — no API key, no network.

Lives in ``graphiti/`` rather than ``dream/`` for the local conftest, which
supplies the ``falkordb_available`` skip guard, the stubbed Graphiti client,
and the no-op ``server`` / ``graph_cleanup`` overrides that keep this suite
off SpinTestServer (postgres + rabbitmq). ``dream/`` has no conftest, and
adding one would change how the existing dream unit tests run.

Run with the platform stack up::

    cd autogpt_platform && docker compose up -d falkordb && cd backend
    poetry run pytest backend/copilot/graphiti/dream_ratification_integration_test.py -xvs
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio

from backend.copilot.dream import apply as apply_mod
from backend.copilot.dream.ratification import run_ratification_pass, try_ratify_on_hit
from backend.copilot.dream.ratification_hits import (
    RATIFICATION_GRACE_PERIOD,
    parse_created_at,
)
from backend.copilot.dream.schemas import ProposedFinding

from . import ingest as ingest_mod
from .client import derive_group_id
from .config import graphiti_config
from .falkordb_driver import AutoGPTFalkorDriver

PASS_ID = "e2e-pass"
SESSION_ID = "e2e-session"
FACT_TEXT = "Atlas deploys are gated on the smoke suite"
INGEST_TIMEOUT_SECONDS = 60.0

PROPOSAL = ProposedFinding(
    content=FACT_TEXT,
    scope="project:atlas",
    confidence=0.62,
    rationale="Two weak-linked episodes both mention the smoke suite blocking a deploy.",
)


# --- Scripted LLM output ----------------------------------------------------

# ``entity_type_id: 0`` is graphiti's built-in 'Entity' type, so both nodes
# carry only the base label — the case the ('Entity','Entity') wildcard in
# ``EDGE_TYPE_MAP`` exists to cover (the #13389 bug). ``relation_type`` must
# name a key of ``EDGE_TYPES`` or graphiti skips edge-attribute extraction
# altogether and the stamp would have nothing to correct.
_SCRIPTED_RESPONSES: dict[str, dict] = {
    "ExtractedEntities": {
        "extracted_entities": [
            {"name": "Atlas", "entity_type_id": 0},
            {"name": "smoke suite", "entity_type_id": 0},
        ]
    },
    "NodeResolutions": {
        "entity_resolutions": [
            {"id": 0, "name": "Atlas", "duplicate_name": ""},
            {"id": 1, "name": "smoke suite", "duplicate_name": ""},
        ]
    },
    "ExtractedEdges": {
        "edges": [
            {
                "source_entity_name": "Atlas",
                "target_entity_name": "smoke suite",
                "relation_type": "MemoryFact",
                "fact": FACT_TEXT,
                "valid_at": None,
                "invalid_at": None,
            }
        ]
    },
    "EdgeDuplicate": {"duplicate_facts": [], "contradicted_facts": []},
    "SummarizedEntities": {"summaries": []},
    # The MemoryFact attribute extractor only ever sees the fact TEXT, so in
    # production it returns the model defaults — which contradict the dream
    # envelope on every field that matters. Reproducing that here is what
    # makes the stamp assertions below meaningful: every value the tests
    # assert is one this payload got wrong.
    "MemoryFact": {
        "status": "active",
        "source_kind": "user_asserted",
        "scope": "real:global",
        "confidence": None,
        "provenance": None,
    },
}


class _FakeRedis:
    """In-memory stand-in for the ``mem:hits:*`` counters.

    The hit counter is best-effort by design (``record_memory_hit`` swallows
    every failure), so pointing it at the dev Redis cluster would only add
    flakiness. What these tests pin is the Cypher status transition each hit
    count drives.
    """

    def __init__(self) -> None:
        self.hits: dict[str, int] = {}

    async def get(self, key: str):
        value = self.hits.get(key, 0)
        return str(value).encode() if value else None

    async def set(self, key: str, value, **kwargs) -> bool:
        self.hits.setdefault(key, int(value))
        return True

    async def incr(self, key: str) -> int:
        self.hits[key] = self.hits.get(key, 0) + 1
        return self.hits[key]

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        return True


# --- Fixtures ---------------------------------------------------------------


@pytest_asyncio.fixture(loop_scope="function")
async def dream_graph(
    falkordb_available: bool,
) -> AsyncIterator[tuple[AutoGPTFalkorDriver, str]]:
    """A per-test FalkorDB database named ``derive_group_id(user_id)``.

    Deliberately not the shared ``clean_graph`` fixture: that one mints a
    standalone ``test_*`` database, while ratification opens its OWN driver
    from the user_id and would read an empty graph. Yields ``(driver,
    user_id)``; everything is detach-deleted afterwards.
    """
    user_id = f"test-{uuid.uuid4().hex[:16]}"
    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=derive_group_id(user_id),
        build_indices=False,
    )
    try:
        yield driver, user_id
    finally:
        try:
            await driver.execute_query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass
        await driver.close()


@pytest.fixture(autouse=True)
def stub_boundaries(
    mocker, dream_graph: tuple[AutoGPTFalkorDriver, str], stub_graphiti_client
):
    """Replace the three external boundaries ingestion would otherwise reach.

    * ``get_graphiti_client`` — patched on ``ingest`` (where it is used) to
      return a Graphiti wired to the test's real driver with the LLM stubbed.
    * ``ensure_dream_system_scheduled`` — ``_ensure_worker`` imports it lazily
      from ``dream.scheduling``, so that module IS the use site. Left live it
      would fire LaunchDarkly + scheduler RPCs from a background task.
    * ``get_redis_async`` — likewise imported lazily by the hit tracker.
    """
    driver, _user_id = dream_graph
    client = stub_graphiti_client(driver, _SCRIPTED_RESPONSES)
    mocker.patch.object(
        ingest_mod, "get_graphiti_client", mocker.AsyncMock(return_value=client)
    )
    mocker.patch(
        "backend.copilot.dream.scheduling.ensure_dream_system_scheduled",
        mocker.AsyncMock(return_value={}),
    )
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        mocker.AsyncMock(return_value=_FakeRedis()),
    )
    return client


@pytest_asyncio.fixture(loop_scope="function", autouse=True)
async def cancel_ingestion_workers() -> AsyncIterator[None]:
    """Stop the per-user ingestion worker before the test's loop closes.

    ``_ensure_worker`` spawns a task that idles for 60s waiting for more
    episodes. Left alone it outlives the test and pytest reports "Task was
    destroyed but it is pending". Reaching into ``_get_loop_state`` is the
    only handle on it — the registry is loop-local by design.
    """
    yield
    state = ingest_mod._get_loop_state()
    workers = list(state.user_workers.values())
    for worker in workers:
        worker.cancel()
    if workers:
        await asyncio.gather(*workers, return_exceptions=True)


# --- Helpers ----------------------------------------------------------------


async def _ingest_dream_proposal(user_id: str) -> None:
    """Write one dream proposal through the production path and await it."""
    completion = ingest_mod.IngestionCompletion()
    queued = await apply_mod._write_proposed_finding(
        user_id,
        PASS_ID,
        0,
        PROPOSAL,
        session_id=SESSION_ID,
        completion=completion,
    )
    assert queued, "proposal was dropped before reaching the ingestion queue"
    completion.register()
    drained = await ingest_mod.wait_for_ingestion(
        completion, timeout_seconds=INGEST_TIMEOUT_SECONDS
    )
    assert drained, f"ingestion did not finish within {INGEST_TIMEOUT_SECONDS}s"


async def _read_edges(driver: AutoGPTFalkorDriver) -> list[dict[str, Any]]:
    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        RETURN e.uuid AS uuid,
               e.status AS status,
               e.source_kind AS source_kind,
               e.scope AS scope,
               e.confidence AS confidence,
               e.provenance AS provenance,
               e.created_at AS created_at,
               e.ratified_at AS ratified_at,
               e.expired_at AS expired_at,
               e.expiration_reason AS expiration_reason
        """
    )
    return records


async def _sole_edge(driver: AutoGPTFalkorDriver) -> dict[str, Any]:
    edges = await _read_edges(driver)
    assert len(edges) == 1, (
        f"expected exactly one :RELATES_TO edge, got {len(edges)}. Zero edges "
        "usually means add_episode raised — the ingestion worker swallows and "
        "logs it, so re-run with --log-cli-level=WARNING to see the cause."
    )
    return edges[0]


async def _backdate_edge(
    driver: AutoGPTFalkorDriver, edge_uuid: str, age: timedelta
) -> None:
    """Age an edge so the sweep's grace-period branch becomes reachable.

    Written as an ISO-8601 string, which is how graphiti's FalkorDB driver
    stores ``created_at`` — ``parse_created_at`` therefore sees the same shape
    it sees in production.
    """
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e.created_at = $created_at",
        uuid=edge_uuid,
        created_at=(datetime.now(timezone.utc) - age).isoformat(),
    )


# --- Tests ------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dream_envelope_lands_tentative_edge_with_envelope_metadata(
    dream_graph,
) -> None:
    """The core regression: a dream proposal's envelope metadata reaches the edge.

    Every asserted value contradicts what the LLM attribute extractor returned
    (see ``_SCRIPTED_RESPONSES['MemoryFact']``), so this fails loudly if the
    deterministic stamp ever goes back to being a no-op.
    """
    driver, user_id = dream_graph
    await _ingest_dream_proposal(user_id)

    edge = await _sole_edge(driver)
    assert edge["status"] == "tentative", (
        "dream proposals must land on probation; 'active' here means the "
        "envelope's status was lost and ratification will never examine it"
    )
    assert edge["source_kind"] == "assistant_derived", (
        "'user_asserted' would grant a dream-derived fact the user-level trust "
        "ratification and retrieval key on"
    )
    assert edge["scope"] == PROPOSAL.scope
    assert edge["confidence"] == pytest.approx(PROPOSAL.confidence)
    assert edge["provenance"].startswith(f"dream:{PASS_ID}:recombine:")
    assert parse_created_at(edge["created_at"]) is not None, (
        "the sweep dates every tentative edge off created_at; a value it cannot "
        "parse leaves the edge tentative forever"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_warm_context_hit_promotes_the_tentative_edge(dream_graph) -> None:
    """Promote-on-hit leg: retrieval flips the freshly ingested edge to active."""
    driver, user_id = dream_graph
    await _ingest_dream_proposal(user_id)
    edge_uuid = (await _sole_edge(driver))["uuid"]

    assert await try_ratify_on_hit(user_id, [edge_uuid]) == 1

    edge = await _sole_edge(driver)
    assert edge["status"] == "active"
    assert edge["ratified_at"] is not None
    assert edge["source_kind"] == "assistant_derived", (
        "promotion must not disturb provenance — an active edge still has to "
        "be attributable to the dream pass that proposed it"
    )

    assert (
        await try_ratify_on_hit(user_id, [edge_uuid]) == 0
    ), "the status='tentative' guard makes repeat hits no-ops"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sweep_supersedes_unhit_edge_past_its_grace_period(dream_graph) -> None:
    """Supersede leg: the nightly sweep retires a tentative edge nobody used."""
    driver, user_id = dream_graph
    await _ingest_dream_proposal(user_id)
    edge_uuid = (await _sole_edge(driver))["uuid"]
    await _backdate_edge(
        driver, edge_uuid, RATIFICATION_GRACE_PERIOD + timedelta(days=1)
    )

    result = await run_ratification_pass(user_id)

    assert result.error is None
    assert result.per_edge_errors == []
    assert result.examined_count == 1, (
        "the sweep lists tentatives with Cypher; examining zero means the edge "
        "never carried status='tentative' in the first place"
    )
    assert result.superseded_count == 1
    assert result.ratified_count == 0

    edge = await _sole_edge(driver)
    assert edge["status"] == "superseded"
    assert edge["expiration_reason"] == "unratified"
    assert edge["expired_at"] is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sweep_leaves_a_fresh_unhit_edge_tentative(dream_graph) -> None:
    """Within the grace period an unhit edge is still earning its keep."""
    driver, user_id = dream_graph
    await _ingest_dream_proposal(user_id)

    result = await run_ratification_pass(user_id)

    assert result.error is None
    assert result.examined_count == 1
    assert result.superseded_count == 0
    assert result.ratified_count == 0
    assert (await _sole_edge(driver))["status"] == "tentative"
