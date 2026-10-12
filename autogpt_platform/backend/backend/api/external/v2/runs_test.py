"""Writes to a run act only on the caller's own run, and sharing goes through
the same service as the web app, so a revoked share can't keep its files
downloadable."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_mock
from fastapi import HTTPException
from prisma.enums import APIKeyPermission

from backend.data import execution as execution_db
from backend.data.execution import ExecutionStatus, GraphExecution
from backend.util.exceptions import NotAuthorizedError

from .models import AgentRunReviewDecision, AgentRunReviewsSubmitRequest
from .runs import delete_run, disable_sharing, enable_sharing, stop_run, submit_reviews
from .tenancy import TenantContext

USER_ID = "user-1"
ORG_ID = "org-1"


def _auth() -> TenantContext:
    return TenantContext(
        user_id=USER_ID,
        scopes=list(APIKeyPermission),
        type="api_key",
        organization_id=ORG_ID,
    )


def _run(status: ExecutionStatus = ExecutionStatus.RUNNING) -> GraphExecution:
    now = datetime.now(timezone.utc)
    return GraphExecution(
        id="run-1",
        user_id=USER_ID,
        graph_id="graph-1",
        graph_version=1,
        inputs={},
        credential_inputs={},
        nodes_input_masks={},
        preset_id=None,
        status=status,
        started_at=now,
        ended_at=None,
        stats=None,
        outputs={},
        organization_id=ORG_ID,
    )


@pytest.fixture
def run_db(mocker: pytest_mock.MockFixture) -> SimpleNamespace:
    """The run lookups and writes; `calls` records the order of the writes."""
    calls: list[str] = []

    def recorder(name: str) -> AsyncMock:
        return AsyncMock(side_effect=lambda **_: calls.append(name))

    db = SimpleNamespace(
        calls=calls,
        get_graph_execution=AsyncMock(return_value=_run()),
        delete_shared_execution_files=recorder("delete_files"),
        update_graph_execution_share_status=recorder("update_share"),
        create_shared_execution_files=recorder("create_files"),
        delete_graph_execution=recorder("delete_run"),
    )
    for name, value in vars(db).items():
        if name != "calls":
            mocker.patch.object(execution_db, name, new=value)
    return db


async def test_sharing_rebuilds_the_file_allowlist(run_db) -> None:
    response = await enable_sharing(run_id="run-1", auth=_auth())

    assert run_db.calls == ["delete_files", "update_share", "create_files"]
    assert response.share_url.endswith(response.share_token)


async def test_unsharing_revokes_the_file_downloads(run_db) -> None:
    await disable_sharing(run_id="run-1", auth=_auth())

    assert run_db.calls == ["delete_files", "update_share"]
    assert (
        run_db.update_graph_execution_share_status.await_args.kwargs["is_shared"]
        is False
    )


async def test_deleting_a_run_revokes_its_file_downloads(run_db) -> None:
    await delete_run(run_id="run-1", auth=_auth())

    assert run_db.calls == ["delete_run", "delete_files"]


@pytest.mark.parametrize(
    "write",
    [
        lambda: enable_sharing(run_id="run-1", auth=_auth()),
        lambda: disable_sharing(run_id="run-1", auth=_auth()),
        lambda: delete_run(run_id="run-1", auth=_auth()),
        lambda: stop_run(run_id="run-1", auth=_auth()),
        lambda: submit_reviews(
            AgentRunReviewsSubmitRequest(
                reviews=[AgentRunReviewDecision(node_exec_id="node-1", approved=True)]
            ),
            run_id="run-1",
            auth=_auth(),
        ),
    ],
    ids=["share", "unshare", "delete", "stop", "review"],
)
async def test_writes_refuse_a_teammates_run(
    run_db, mocker: pytest_mock.MockFixture, write
) -> None:
    """A teammate's run in the same organization is visible to reads, but the
    writes beneath these routes are its owner's: none happens, and the caller
    gets 403."""
    run_db.get_graph_execution.return_value = _run().model_copy(
        update={"user_id": "teammate"}
    )
    stop = mocker.patch(
        "backend.executor.utils.stop_graph_execution", new_callable=AsyncMock
    )
    reviews = mocker.patch(
        "backend.api.external.v2.runs.process_reviews", new_callable=AsyncMock
    )

    with pytest.raises(NotAuthorizedError):
        await write()

    assert run_db.calls == []
    stop.assert_not_awaited()
    reviews.assert_not_awaited()


@pytest.mark.parametrize(
    "finished",
    [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.TERMINATED],
)
async def test_stopping_a_finished_run_is_a_conflict(
    run_db, mocker: pytest_mock.MockFixture, finished: ExecutionStatus
) -> None:
    run_db.get_graph_execution.return_value = _run(finished)
    stop = mocker.patch(
        "backend.executor.utils.stop_graph_execution", new_callable=AsyncMock
    )

    with pytest.raises(HTTPException) as caught:
        await stop_run(run_id="run-1", auth=_auth())

    assert caught.value.status_code == 409
    stop.assert_not_awaited()


@pytest.mark.parametrize(
    "stoppable",
    [
        ExecutionStatus.INCOMPLETE,
        ExecutionStatus.QUEUED,
        ExecutionStatus.RUNNING,
        ExecutionStatus.REVIEW,
    ],
)
async def test_stopping_an_unfinished_run_returns_its_status(
    run_db, mocker: pytest_mock.MockFixture, stoppable: ExecutionStatus
) -> None:
    run_db.get_graph_execution.side_effect = [
        _run(stoppable),
        _run(ExecutionStatus.TERMINATED),
    ]
    stop = mocker.patch(
        "backend.executor.utils.stop_graph_execution", new_callable=AsyncMock
    )

    run = await stop_run(run_id="run-1", auth=_auth())

    assert run.status == "TERMINATED"
    assert stop.await_args.kwargs["graph_exec_id"] == "run-1"


async def test_a_slow_stop_returns_the_run_instead_of_failing(
    run_db, mocker: pytest_mock.MockFixture
) -> None:
    """The cancel is already published when the wait times out; the caller
    gets the run as it stands, not a 500."""
    run_db.get_graph_execution.side_effect = [_run(), _run()]
    mocker.patch(
        "backend.executor.utils.stop_graph_execution",
        new=AsyncMock(side_effect=TimeoutError()),
    )

    run = await stop_run(run_id="run-1", auth=_auth())

    assert run.status == "RUNNING"
