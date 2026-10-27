"""A member removed from an org has the runs they started there stopped."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from autogpt_libs.auth.models import RequestContext

from backend.api.features.orgs import member_runs, routes
from backend.data.execution import ExecutionStatus

ORG_ID = "org-1"
MEMBER_ID = "member-1"


def _admin_ctx() -> RequestContext:
    return RequestContext(
        user_id="admin-1",
        org_id=ORG_ID,
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


@pytest.fixture
def steps(mocker):
    """Patches removal, the run lookups and stopping a run, recording the order
    they happen in. The member has two runs going in the org, and a third they
    deleted from their history."""
    order: list[str] = []
    org_db = mocker.patch("backend.api.features.orgs.routes.org_db")
    org_db.remove_org_member = AsyncMock(
        side_effect=lambda *_, **__: order.append("removed")
    )
    runs = {
        False: [SimpleNamespace(id="run-1"), SimpleNamespace(id="run-2")],
        True: [SimpleNamespace(id="deleted-run")],
    }
    execution_db = mocker.patch("backend.api.features.orgs.member_runs.execution_db")
    execution_db.get_graph_executions = AsyncMock(
        side_effect=lambda **kwargs: order.append("looked up")
        or runs[kwargs["deleted"]]
    )
    stop = mocker.patch(
        "backend.api.features.orgs.member_runs.stop_graph_execution",
        new=AsyncMock(),
    )
    return SimpleNamespace(
        order=order,
        remove=org_db.remove_org_member,
        lookup=execution_db.get_graph_executions,
        stop=stop,
    )


def _waits(stop: AsyncMock) -> dict[str, float]:
    """How long each stopped run's stop waited for it to end, by run id."""
    calls = stop.await_args_list
    assert {call.kwargs["user_id"] for call in calls} == {MEMBER_ID}
    return {call.kwargs["graph_exec_id"]: call.kwargs["wait_timeout"] for call in calls}


@pytest.mark.asyncio
async def test_removing_a_member_stops_their_unfinished_runs_in_the_org(steps):
    await routes.remove_member(org_id=ORG_ID, uid=MEMBER_ID, ctx=_admin_ctx())

    assert steps.order == ["removed", "looked up", "looked up"]
    for call in steps.lookup.await_args_list:
        lookup = call.kwargs
        assert (lookup["user_id"], lookup["organization_id"]) == (MEMBER_ID, ORG_ID)
        assert set(lookup["statuses"]) == {
            ExecutionStatus.QUEUED,
            ExecutionStatus.INCOMPLETE,
            ExecutionStatus.RUNNING,
            ExecutionStatus.REVIEW,
        }
    waits = _waits(steps.stop)
    assert (waits["run-1"], waits["run-2"]) == (15.0, 15.0)


@pytest.mark.asyncio
async def test_a_run_deleted_from_history_is_stopped_too(steps):
    """Deleting a run hides it without stopping it. Lookups can't see it, so
    its stop sends the cancel without waiting to see it end."""
    await routes.remove_member(org_id=ORG_ID, uid=MEMBER_ID, ctx=_admin_ctx())

    assert {call.kwargs["deleted"] for call in steps.lookup.await_args_list} == {
        False,
        True,
    }
    assert _waits(steps.stop)["deleted-run"] == 0


@pytest.mark.asyncio
async def test_a_run_that_wont_stop_doesnt_fail_the_removal(steps, caplog):
    """The member is already out of the org; the other runs are still stopped,
    and the one that wasn't is logged."""

    async def stop(user_id: str, graph_exec_id: str, wait_timeout: float) -> None:
        if graph_exec_id == "run-1":
            raise TimeoutError("still stopping")

    steps.stop.side_effect = stop

    with caplog.at_level(logging.WARNING, logger=member_runs.__name__):
        await routes.remove_member(org_id=ORG_ID, uid=MEMBER_ID, ctx=_admin_ctx())

    assert set(_waits(steps.stop)) == {"run-1", "run-2", "deleted-run"}
    assert [
        record.levelno for record in caplog.records if "run-1" in record.message
    ] == [logging.WARNING]


@pytest.mark.parametrize(
    "failing, still_stopped",
    [(False, {"deleted-run"}), (True, {"run-1", "run-2"})],
    ids=["visible-lookup-fails", "deleted-lookup-fails"],
)
@pytest.mark.asyncio
async def test_a_failed_lookup_doesnt_fail_the_removal(
    steps, caplog, failing: bool, still_stopped: set[str]
):
    """The member is already out of the org, so the request succeeds, the
    failure is logged, and the runs the other lookup found are still stopped.
    Retrying the request would find no member to remove."""
    find = steps.lookup.side_effect

    def lookup(**kwargs):
        if kwargs["deleted"] is failing:
            raise ConnectionError("database unavailable")
        return find(**kwargs)

    steps.lookup.side_effect = lookup

    with caplog.at_level(logging.WARNING, logger=member_runs.__name__):
        await routes.remove_member(org_id=ORG_ID, uid=MEMBER_ID, ctx=_admin_ctx())

    steps.remove.assert_awaited_once()
    assert set(_waits(steps.stop)) == still_stopped
    assert [
        record.levelno for record in caplog.records if MEMBER_ID in record.message
    ] == [logging.WARNING]


@pytest.mark.asyncio
async def test_a_member_who_isnt_removed_keeps_their_runs(steps):
    steps.remove.side_effect = ValueError("Cannot remove the org owner.")

    with pytest.raises(ValueError):
        await routes.remove_member(org_id=ORG_ID, uid=MEMBER_ID, ctx=_admin_ctx())

    steps.lookup.assert_not_awaited()
    steps.stop.assert_not_awaited()
