"""A forget reaches what the dream derived from the fact it forgot, on a live
FalkorDB, the dream's facts written through ``dream/apply.py`` and the
production ingestion worker (``recall_cascade.py``).

The bakery (``recall_cascade_fixtures.py``): the user said Sunrise Bakery
uses Hill Country Mills as its flour supplier, and three dream passes derived
a consolidation from it, a proposal from that and from a fact the user
stated about the same bakery, and a proposal from the consolidation's dream
text. On the test bed a forget left 16 such facts live, and the assistant
went on answering with the forgotten one. A hard forget's erasure is in
``recall_cascade_erase_integration_test.py``.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_cascade_integration_test.py
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from backend.copilot.dream import apply
from backend.copilot.dream.schemas import (
    ConsolidatedFact,
    DreamOperations,
    ProposedFinding,
)
from backend.copilot.model import ChatSession
from backend.copilot.tools import graphiti_search
from backend.copilot.tools.models import MemorySearchResponse

from . import context
from .falkordb_driver import AutoGPTFalkorDriver
from .recall import FORGOTTEN_FACT
from .recall_cascade_fixtures import (
    BOULE,
    BOULE_FLOUR,
    FLOUR,
    SUPPLIES,
    WEEKLY,
    Bakery,
    build_bakery,
    dream,
    dream_episode,
    dream_once,
    fact_uuid,
    gather,
)
from .recall_forget import retract
from .recall_integration_fixtures import (
    Fact,
    capture_spawned_tasks,
    edge_row,
    episode_row,
    live_facts,
    patch_recall_boundaries,
    stop_ingestion_workers,
)
from .recall_stamp import stamp_recalls
from .scope import MemoryScope

_DERIVED = [SUPPLIES[2], BOULE_FLOUR[2], WEEKLY[2]]
_DEPENDS: Fact = (
    "Sourdough Boule",
    "Hill Country Mills",
    "The sourdough boule depends on Hill Country Mills",
)


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest_asyncio.fixture(loop_scope="function")
async def dream_apply(mocker: MockerFixture) -> AsyncIterator[None]:
    """apply's Postgres side (the dream session and its summary) stubbed."""
    mocker.patch.object(apply, "_create_dream_session", AsyncMock(return_value="s"))
    mocker.patch.object(apply, "_write_dream_summary_message", AsyncMock())
    mocker.patch(
        "backend.copilot.dream.registry.ensure_dream_system_scheduled",
        AsyncMock(return_value=None),
    )
    yield
    await stop_ingestion_workers()


@pytest.fixture
def hit_tasks(mocker) -> list[asyncio.Task]:
    return capture_spawned_tasks(mocker)


async def _shown(scope: MemoryScope, hit_tasks: list[asyncio.Task]) -> str:
    """Everything the assistant is shown for the bakery's flour: warm
    context, and ``memory_search``'s facts and episodes."""
    query = "Sunrise Bakery flour Hill Country Mills"
    warm = await context._fetch(scope, query)
    await asyncio.gather(*context._pending_hit_tasks)
    session = ChatSession.new(scope.owner_user_id, dry_run=False)
    found = await graphiti_search.MemorySearchTool()._execute(
        scope.owner_user_id, session, query=query
    )
    await asyncio.gather(*hit_tasks)
    assert isinstance(found, MemorySearchResponse)
    return "\n".join([warm or "", *found.facts, *found.recent_episodes])


async def _set_status(
    driver: AutoGPTFalkorDriver, uuid: str, status: str, reason: str | None = None
) -> None:
    """As ratification or a dream demotion leaves a fact (expired with a
    reason, when ``reason`` is given)."""
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() "
        "SET e.status = $status, e.expiration_reason = $reason, "
        "e.expired_at = CASE WHEN $reason IS NULL THEN NULL ELSE $now END",
        uuid=uuid,
        status=status,
        reason=reason,
        now=datetime.now(timezone.utc).isoformat(),
    )


async def _assert_retracted_for(
    driver: AutoGPTFalkorDriver, bakery: Bakery, sentences: dict[str, str]
) -> None:
    """Each derived fact retracted as a soft forget retracts one, naming the
    forgotten flour supplier, its sentence kept only as the audit copy."""
    for uuid, sentence in sentences.items():
        row = await edge_row(driver, uuid)
        assert row["forgotten_at"] is not None, sentence
        assert row["status"] == "retracted"
        assert row["reason"] == f"derived_from_forgotten:{bakery.flour}"
        assert (row["fact"], row["fact_redacted"]) == (FORGOTTEN_FACT, sentence)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_retracts_everything_the_dream_derived_from_the_fact(
    scope_graph, stub_graphiti_client, dream_apply, hit_tasks
) -> None:
    driver, scope = scope_graph
    bakery = await build_bakery(driver, scope, stub_graphiti_client)
    before = await _shown(scope, hit_tasks)
    assert all(sentence in before for sentence in [FLOUR[2], *_DERIVED])

    result = await retract(scope, [bakery.flour])

    assert (result.deleted, result.failures) == ([bakery.flour], [])
    assert sorted(result.derived) == bakery.derived()
    assert set(await live_facts(driver)) == {bakery.boule, bakery.cafe}
    derived = [bakery.supplies, bakery.boule_flour, bakery.weekly]
    await _assert_retracted_for(driver, bakery, dict(zip(derived, _DERIVED)))
    for episode in (
        bakery.said,
        bakery.supplies_episode,
        bakery.boule_flour_episode,
        bakery.weekly_episode,
    ):
        assert (await episode_row(driver, episode))["redacted_at"] is not None
    after = await _shown(scope, hit_tasks)
    assert not [s for s in [FLOUR[2], *_DERIVED] if s in after], "still shown"
    assert BOULE[2] in after, "the bakery fact the user stated stays"
    assert await stamp_recalls(driver, derived, owner="test") == 0, "never stamped"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_next_dream_pass_cannot_teach_them_again(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    """A pass reading after the forget sees none of it; one that read before
    has its writes resting on any of it dropped under the write lock."""
    driver, scope = scope_graph
    bakery = await build_bakery(driver, scope, stub_graphiti_client)
    stale = await gather(scope)

    await retract(scope, [bakery.flour])

    fresh = await gather(scope)
    assert {fact.uuid for fact in fresh.facts} == {bakery.boule, bakery.cafe}
    hidden = {
        bakery.said,
        bakery.supplies_episode,
        bakery.boule_flour_episode,
        bakery.weekly_episode,
    }
    assert not hidden & {episode.uuid for episode in fresh.episodes}
    ops = DreamOperations(
        writes=[
            ConsolidatedFact(
                content=SUPPLIES[2],
                confidence=0.9,
                source_fact_uuids=[bakery.supplies],
            )
        ],
        proposals=[
            ProposedFinding(
                content=WEEKLY[2],
                confidence=0.6,
                rationale="read before the forget",
                source_episode_uuids=[bakery.weekly_episode],
            )
        ],
    )
    live = await live_facts(driver)
    stats = await dream(
        driver, scope, stub_graphiti_client, ops, stale, SUPPLIES, "pass-late"
    )
    assert (stats["dropped_forgotten"], stats["uncited_writes_dropped"]) == (2, 0)
    assert await live_facts(driver) == live


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_walk_goes_on_through_a_derived_fact_no_longer_live(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    """The proposal ``boule_flour`` was ratified, a later pass derived a fact
    from it, and then the dream superseded it: the proposal keeps its status,
    but the walk goes on through it, retracting the fact derived from it and
    hiding both dream texts."""
    driver, scope = scope_graph
    bakery = await build_bakery(driver, scope, stub_graphiti_client)
    await _set_status(driver, bakery.boule_flour, "active")
    later = ProposedFinding(
        content=_DEPENDS[2],
        confidence=0.6,
        rationale="the ratified proposal",
        source_fact_uuids=[bakery.boule_flour],
    )
    await dream_once(driver, scope, stub_graphiti_client, later, _DEPENDS, "pass-4")
    depends = await fact_uuid(driver, _DEPENDS[2])
    await _set_status(driver, bakery.boule_flour, "superseded", "stale_fact")

    result = await retract(scope, [bakery.flour])

    assert sorted(result.derived) == sorted([bakery.supplies, bakery.weekly, depends])
    row = await edge_row(driver, bakery.boule_flour)
    assert (row["status"], row["reason"]) == ("superseded", "stale_fact")
    episodes = (bakery.boule_flour_episode, await dream_episode(driver, "pass-4"))
    for episode in episodes:
        assert (await episode_row(driver, episode))["redacted_at"] is not None
    assert set(await live_facts(driver)) == {bakery.boule, bakery.cafe}
