"""The recall guard inside apply: the demotions' recall stamps are read from
the graph again under the pass's lease, right before the writes, so a fact
recalled after the pass gathered its input is left alone; entity
invalidations skip their protected neighbours; every drop is counted."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.graphiti.recall_stamp import RecallStamp, stamp_time
from backend.copilot.graphiti.scope import MemoryScope

from . import apply as apply_mod
from . import recall_guard
from .schemas import (
    DreamDemotion,
    DreamOperations,
    DreamOperationsSnapshot,
    EntityInvalidation,
)

_SCOPE = MemoryScope.for_user("u-guard")


def _recalled_now(uuid: str) -> RecallStamp:
    return RecallStamp(
        uuid=uuid,
        recall_count=4,
        last_recalled_at=stamp_time(datetime.now(timezone.utc)),
    )


@pytest.fixture
def graph(mocker) -> SimpleNamespace:
    """apply's graph and chat boundaries, recording what reaches them. The
    guard's reads answer ``stamps`` / ``neighbours`` (nothing recalled until
    a test says otherwise)."""
    calls: list[str] = []
    state = SimpleNamespace(calls=calls, stamps=[], neighbours=[])

    async def read(driver, group_id, uuids):
        calls.append("reread")
        return [s for s in state.stamps if s.uuid in uuids]

    async def read_neighbours(driver, group_id, entity_uuid):
        calls.append("neighbours")
        return state.neighbours

    async def supersede(driver, uuids, **kwargs):
        calls.append("write")
        return list(uuids), []

    driver = MagicMock()
    driver.close = AsyncMock()
    mocker.patch.object(apply_mod, "open_driver", return_value=driver)
    state.supersede = mocker.patch.object(
        apply_mod, "mark_edges_superseded", AsyncMock(side_effect=supersede)
    )
    state.invalidate = mocker.patch.object(
        apply_mod,
        "invalidate_entity_direct_neighbors",
        AsyncMock(return_value=["plain"]),
    )
    mocker.patch.object(recall_guard, "read_recall_stamps", side_effect=read)
    mocker.patch.object(
        recall_guard, "read_neighbour_stamps", side_effect=read_neighbours
    )
    mocker.patch.object(apply_mod, "is_feature_enabled", AsyncMock(return_value=True))
    database = MagicMock()
    database.create_chat_session = AsyncMock()
    database.update_chat_session_title = AsyncMock()
    database.add_chat_message = AsyncMock()
    mocker.patch("backend.data.db_accessors.chat_db", return_value=database)
    mocker.patch(
        "backend.api.features.orgs.db.get_user_default_team",
        AsyncMock(return_value=(None, None)),
    )
    return state


def _demotions(*uuids: str) -> list[DreamDemotion]:
    return [DreamDemotion(edge_uuid=uuid, reason="stale_fact") for uuid in uuids]


def _written(graph: SimpleNamespace) -> list[str]:
    return [u for call in graph.supersede.await_args_list for u in call.args[1]]


@pytest.mark.asyncio
async def test_a_fact_recalled_after_the_gather_is_not_demoted(graph) -> None:
    graph.stamps = [_recalled_now("late")]
    ops = DreamOperations(demotions=_demotions("late", "cold"), summary_for_user="x")

    stats = await apply_mod.apply_operations(
        _SCOPE,
        "p-late",
        ops,
        known_fact_uuids={"late", "cold"},
        protected_demotions=2,
    )

    assert _written(graph) == ["cold"]
    assert stats["demotion_count"] == 1
    # Two dropped at clamp time, one here.
    assert stats["protected_demotions"] == 3
    snapshot = stats["snapshot"]
    assert isinstance(snapshot, DreamOperationsSnapshot)
    assert [d.edge_uuid for d in snapshot.demotions] == ["cold"]


@pytest.mark.asyncio
async def test_the_reread_runs_under_the_lease_right_before_the_writes(
    graph,
) -> None:
    lease = MagicMock()

    async def renew() -> bool:
        graph.calls.append("renew")
        return True

    lease.renew = AsyncMock(side_effect=renew)
    ops = DreamOperations(demotions=_demotions("cold"), summary_for_user="x")

    await apply_mod.apply_operations(
        _SCOPE, "p-order", ops, known_fact_uuids={"cold"}, lease=lease
    )

    assert graph.calls == ["renew", "reread", "write"]


@pytest.mark.asyncio
async def test_an_override_still_demotes_a_fact_recalled_after_the_gather(
    graph,
) -> None:
    graph.stamps = [_recalled_now("wrong"), _recalled_now("contradicted")]
    ops = DreamOperations(
        demotions=[
            DreamDemotion(edge_uuid="wrong", reason="user_signal"),
            DreamDemotion(edge_uuid="contradicted", reason="contradicted_by:witness"),
        ],
        summary_for_user="x",
    )

    stats = await apply_mod.apply_operations(
        _SCOPE,
        "p-override",
        ops,
        known_fact_uuids={"wrong", "contradicted", "witness"},
    )

    assert sorted(_written(graph)) == ["contradicted", "wrong"]
    assert stats["protected_demotions"] == 0


@pytest.mark.asyncio
async def test_an_entity_invalidation_skips_and_counts_protected_neighbours(
    graph,
) -> None:
    graph.neighbours = [_recalled_now("recalled"), RecallStamp(uuid="plain")]
    ops = DreamOperations(
        entity_invalidations=[EntityInvalidation(entity_uuid="hub", reason="gone")],
        summary_for_user="x",
    )

    stats = await apply_mod.apply_operations(_SCOPE, "p-hub", ops)

    assert graph.invalidate.await_args.kwargs["skip"] == {"recalled"}
    snapshot = stats["snapshot"]
    assert isinstance(snapshot, DreamOperationsSnapshot)
    [summary] = snapshot.entity_invalidations
    assert (summary.edges_touched, summary.edges_protected) == (["plain"], ["recalled"])
    assert stats["protected_demotions"] == 1


@pytest.mark.asyncio
async def test_a_pass_left_empty_by_the_clamp_still_reports_its_count(
    graph,
) -> None:
    stats = await apply_mod.apply_operations(
        _SCOPE,
        "p-empty",
        DreamOperations(summary_for_user="nothing"),
        protected_demotions=2,
    )

    assert "session_id" not in stats
    assert stats["protected_demotions"] == 2
    assert graph.calls == []
