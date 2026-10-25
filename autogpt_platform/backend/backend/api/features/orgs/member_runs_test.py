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
    """Patches removal, the run lookup and stopping a run, recording the order
    they happen in. The member has two runs going in the org."""
    order: list[str] = []
    org_db = mocker.patch("backend.api.features.orgs.routes.org_db")
    org_db.remove_org_member = AsyncMock(
        side_effect=lambda *_, **__: order.append("removed")
    )
    runs = [SimpleNamespace(id="run-1"), SimpleNamespace(id="run-2")]
    execution_db = mocker.patch("backend.api.features.orgs.member_runs.execution_db")
    execution_db.get_graph_executions = AsyncMock(
        side_effect=lambda **_: order.append("looked up") or runs
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


@pytest.mark.asyncio
async def test_removing_a_member_stops_their_unfinished_runs_in_the_org(steps):
    await routes.remove_member(org_id=ORG_ID, uid=MEMBER_ID, ctx=_admin_ctx())

    assert steps.order == ["removed", "looked up"]
    lookup = steps.lookup.await_args.kwargs
    assert (lookup["user_id"], lookup["organization_id"]) == (MEMBER_ID, ORG_ID)
    assert set(lookup["statuses"]) == {
        ExecutionStatus.QUEUED,
        ExecutionStatus.INCOMPLETE,
        ExecutionStatus.RUNNING,
        ExecutionStatus.REVIEW,
    }
    stopped = {
        (call.kwargs["user_id"], call.kwargs["graph_exec_id"])
        for call in steps.stop.await_args_list
    }
    assert stopped == {(MEMBER_ID, "run-1"), (MEMBER_ID, "run-2")}


@pytest.mark.asyncio
async def test_a_run_that_wont_stop_doesnt_fail_the_removal(steps, caplog):
    """The member is already out of the org; the other runs are still stopped,
    and the one that wasn't is logged."""

    async def stop(user_id: str, graph_exec_id: str) -> None:
        if graph_exec_id == "run-1":
            raise TimeoutError("still stopping")

    steps.stop.side_effect = stop

    with caplog.at_level(logging.WARNING, logger=member_runs.__name__):
        await routes.remove_member(org_id=ORG_ID, uid=MEMBER_ID, ctx=_admin_ctx())

    assert {call.kwargs["graph_exec_id"] for call in steps.stop.await_args_list} == {
        "run-1",
        "run-2",
    }
    assert [
        record.levelno for record in caplog.records if "run-1" in record.message
    ] == [logging.WARNING]


@pytest.mark.asyncio
async def test_a_member_who_isnt_removed_keeps_their_runs(steps):
    steps.remove.side_effect = ValueError("Cannot remove the org owner.")

    with pytest.raises(ValueError):
        await routes.remove_member(org_id=ORG_ID, uid=MEMBER_ID, ctx=_admin_ctx())

    steps.lookup.assert_not_awaited()
    steps.stop.assert_not_awaited()
