"""The destructive stage (``demotions.py``) with its two guarded writers
mocked: the protection each write carries for its reason, and that what the
pass reports is what the writes returned. What the statements do is tested
with the writers and on FalkorDB (``graphiti/recall_guard_integration_test.py``);
that usage never raises the count is ``demotions_stateful_test.py``."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.graphiti.guarded_writes import NeighbourWrites, WriteOutcome
from backend.copilot.graphiti.recall_stamp import RecallProtection, parse_stamp
from backend.copilot.graphiti.scope import MemoryScope

from . import demotions as demotions_mod
from . import recall_guard
from .demotions import apply_demotions
from .schemas import (
    DemotionSummary,
    DreamDemotion,
    DreamOperations,
    EntityInvalidation,
    EntityInvalidationSummary,
)

_SCOPE = MemoryScope.for_user("u-stage")
C, S, F = WriteOutcome.CHANGED, WriteOutcome.SPARED, WriteOutcome.FAILED


@pytest.fixture
def stage(mocker) -> SimpleNamespace:
    """The stage's boundaries: its driver, the flag (on), no bundle, a
    30-day window, and the two writers (every demotion lands, nothing
    spared) unless a test says otherwise."""
    driver = MagicMock()
    driver.close = AsyncMock()
    state = SimpleNamespace(
        driver=driver,
        open_driver=mocker.patch.object(
            demotions_mod, "open_driver", return_value=driver
        ),
        supersede=mocker.patch.object(
            demotions_mod,
            "supersede_unless_recalled",
            AsyncMock(side_effect=lambda driver, uuids, **kw: [C] * len(uuids)),
        ),
        invalidate=mocker.patch.object(
            demotions_mod,
            "invalidate_entity_direct_neighbors",
            AsyncMock(return_value=NeighbourWrites()),
        ),
    )
    mocker.patch.object(
        demotions_mod, "is_feature_enabled", AsyncMock(return_value=True)
    )
    mocker.patch.object(
        demotions_mod, "read_input_bundle", AsyncMock(return_value=None)
    )
    set_window(mocker, 30)
    return state


def set_window(mocker, days: int) -> None:
    mocker.patch.object(
        recall_guard,
        "Settings",
        return_value=SimpleNamespace(
            config=SimpleNamespace(dream_demotion_protect_days=days)
        ),
    )


def _demote(uuid: str, reason: str = "stale_fact") -> DreamDemotion:
    return DreamDemotion(edge_uuid=uuid, reason=reason)


def _protections(writer: AsyncMock) -> dict[str, RecallProtection]:
    return {
        call.kwargs["reason"]: call.kwargs["protection"]
        for call in writer.await_args_list
    }


@pytest.mark.asyncio
async def test_each_write_carries_the_protection_for_its_reason(stage) -> None:
    ops = DreamOperations(
        demotions=[
            _demote("a"),
            _demote("b", "user_signal"),
            _demote("c", "contradicted_by:k"),
            _demote("d", "contradicted_by:ghost"),
        ],
        entity_invalidations=[
            EntityInvalidation(entity_uuid="hub", reason="user_signal")
        ],
    )
    before = datetime.now(timezone.utc)

    await apply_demotions(_SCOPE, "p-1", ops, {"a", "b", "c", "d", "k"})

    after = datetime.now(timezone.utc)
    direct = _protections(stage.supersede)
    assert {r: (p.override, p.cited) for r, p in direct.items()} == {
        "stale_fact": (False, None),
        "user_signal": (True, None),
        "contradicted_by:k": (True, "k"),
        "contradicted_by:ghost": (False, "ghost"),
    }
    neighbour = stage.invalidate.await_args.kwargs["protection"]
    assert (neighbour.override, neighbour.cited) == (True, None)
    # One window start for the whole stage: 30 days before it ran.
    starts = {p.recalled_since for p in [*direct.values(), neighbour]}
    assert len(starts) == 1
    start = parse_stamp(starts.pop())
    assert start is not None
    assert before - timedelta(days=30) <= start <= after - timedelta(days=30)


@pytest.mark.asyncio
async def test_what_the_pass_reports_is_what_the_writes_did(stage) -> None:
    stage.supersede.side_effect = None
    stage.supersede.return_value = [C, S, F]
    stage.invalidate.return_value = NeighbourWrites(changed=["x"], spared=["y", "z"])
    ops = DreamOperations(
        demotions=[_demote("a"), _demote("b"), _demote("c")],
        entity_invalidations=[EntityInvalidation(entity_uuid="hub", reason="gone")],
    )

    results = await apply_demotions(_SCOPE, "p-2", ops, {"a", "b", "c"})

    assert results.demotions == [
        DemotionSummary(edge_uuid="a", reason="stale_fact", new_status="superseded"),
        DemotionSummary(
            edge_uuid="b",
            reason="stale_fact",
            new_status="superseded",
            applied=False,
            protected=True,
        ),
        DemotionSummary(
            edge_uuid="c", reason="stale_fact", new_status="superseded", applied=False
        ),
    ]
    assert results.entity_invalidations == [
        EntityInvalidationSummary(
            entity_uuid="hub",
            reason="gone",
            edges_touched=["x"],
            edges_protected=["y", "z"],
        )
    ]
    assert (results.demoted, results.failed, results.entity_edges) == (1, 1, 1)
    assert results.protected == 3, "one spared demotion and two spared neighbours"


@pytest.mark.asyncio
async def test_a_fact_spared_by_several_writes_counts_once(stage) -> None:
    """``[a, a]`` spared twice, and ``a`` spared again as a neighbour: one
    protected fact, plus the other neighbour."""
    stage.supersede.side_effect = None
    stage.supersede.return_value = [S, S]
    stage.invalidate.return_value = NeighbourWrites(spared=["a", "b"])
    ops = DreamOperations(
        demotions=[_demote("a"), _demote("a")],
        entity_invalidations=[EntityInvalidation(entity_uuid="hub", reason="gone")],
    )

    results = await apply_demotions(_SCOPE, "p-8", ops, {"a"})

    assert [d.protected for d in results.demotions] == [True, True]
    assert results.entity_invalidations[0].edges_protected == ["a", "b"]
    assert results.protected == 2


@pytest.mark.asyncio
async def test_a_fact_a_later_write_changes_is_not_counted_as_protected(
    stage,
) -> None:
    """The direct demotion spares ``a``; the user's retraction through its
    entity then changes it. Each operation records its own write, but ``a``
    is not a fact protection kept, so it counts only as changed."""
    stage.supersede.side_effect = None
    stage.supersede.return_value = [S]
    stage.invalidate.return_value = NeighbourWrites(changed=["a"])
    ops = DreamOperations(
        demotions=[_demote("a")],
        entity_invalidations=[
            EntityInvalidation(entity_uuid="hub", reason="user_signal")
        ],
    )

    results = await apply_demotions(_SCOPE, "p-9", ops, {"a"})

    assert results.demotions[0].protected is True
    assert results.entity_invalidations[0].edges_touched == ["a"]
    assert (results.demoted, results.entity_edges, results.protected) == (0, 1, 0)


@pytest.mark.asyncio
async def test_each_demotion_gets_its_own_outcome_in_bucket_order(stage) -> None:
    """Grouped by status and reason as before; a duplicate target and a
    demotion from another bucket keep their own outcomes."""
    outcomes = {"stale_fact": [C, S, F], "user_signal": [C]}
    stage.supersede.side_effect = lambda driver, uuids, **kw: outcomes[kw["reason"]]
    ops = DreamOperations(
        demotions=[
            _demote("a"),
            _demote("b", "user_signal"),
            _demote("c"),
            _demote("a"),
        ]
    )

    results = await apply_demotions(_SCOPE, "p-3", ops, {"a", "b", "c"})

    assert [call.args[1] for call in stage.supersede.await_args_list] == [
        ["a", "c", "a"],
        ["b"],
    ]
    assert [(d.edge_uuid, d.applied, d.protected) for d in results.demotions] == [
        ("a", True, False),
        ("b", True, False),
        ("c", False, True),
        ("a", False, False),
    ]


@pytest.mark.asyncio
async def test_nothing_is_read_before_the_writes(stage) -> None:
    """The protection is in the writes: the stage itself sends the graph
    nothing, so no read of stamps can fail open or go stale."""
    ops = DreamOperations(
        demotions=[_demote("a")],
        entity_invalidations=[EntityInvalidation(entity_uuid="hub", reason="gone")],
    )

    await apply_demotions(_SCOPE, "p-4", ops, {"a"})

    stage.driver.execute_query.assert_not_called()
    stage.open_driver.assert_called_once_with(_SCOPE)
    stage.driver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_zero_window_writes_with_no_protection(stage, mocker) -> None:
    set_window(mocker, 0)
    ops = DreamOperations(demotions=[_demote("a")])

    await apply_demotions(_SCOPE, "p-5", ops, {"a"})

    protection = stage.supersede.await_args.kwargs["protection"]
    assert protection.recalled_since is None


@pytest.mark.asyncio
async def test_without_the_facts_the_pass_read_no_contradiction_overrides(
    stage,
) -> None:
    """No allowlist and no bundle: the demotions are kept, as before, but a
    contradiction can then cite nothing, so it overrides no protection."""
    ops = DreamOperations(demotions=[_demote("a", "contradicted_by:k")])

    await apply_demotions(_SCOPE, "p-6", ops, None)

    assert stage.supersede.await_args.args[1] == ["a"]
    assert stage.supersede.await_args.kwargs["protection"].override is False


@pytest.mark.asyncio
async def test_no_operations_open_no_driver(stage) -> None:
    results = await apply_demotions(_SCOPE, "p-7", DreamOperations(), {"a"})

    assert results.demotions == [] and results.entity_invalidations == []
    stage.open_driver.assert_not_called()
