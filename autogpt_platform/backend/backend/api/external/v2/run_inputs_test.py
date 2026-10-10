"""Run start and schedule create refuse inputs the graph's `input_schema` rejects.

The executor runs a graph with a misspelled or missing input anyway, and the run
ends COMPLETED with no outputs and no error, so the request is the only place a
client can learn its inputs were wrong.
"""

from typing import Any
from unittest import mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from prisma.enums import APIKeyPermission

from backend.blocks.io import AgentInputBlock
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.data.graph import BaseGraph

from .errors import add_v2_exception_handlers
from .idempotency import IDEMPOTENCY_HEADER
from .library import library_router
from .schedules import schedules_router
from .tenancy import TenantContext, require_auth

ORG_ID = "org-1"
INPUT_SCHEMA = BaseGraph._generate_schema(
    (AgentInputBlock.Input, {"name": "topic"}),
    (AgentInputBlock.Input, {"name": "tone", "value": "neutral"}),
)


@pytest.mark.parametrize("path", ["/library/agents/agent-1/runs", "/schedules"])
@pytest.mark.parametrize(
    "inputs,bad_input,error_type",
    [
        ({"topic": "AI", "topik": "AI"}, "topik", "extra_forbidden"),
        ({"tone": "dry"}, "topic", "missing"),
        ({"topic": None}, "topic", "missing"),
    ],
)
def test_inputs_the_graph_rejects_are_a_422_naming_them(
    client: fastapi.testclient.TestClient,
    started: mock.AsyncMock,
    path: str,
    inputs: dict[str, Any],
    bad_input: str,
    error_type: str,
) -> None:
    response = client.post(path, json=_body(path, inputs))

    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert [(e["loc"], e["type"]) for e in error["details"]["errors"]] == [
        (["body", "inputs", bad_input], error_type)
    ]
    started.assert_not_awaited()


def test_inputs_the_graph_accepts_start_the_run(
    client: fastapi.testclient.TestClient, started: mock.AsyncMock
) -> None:
    response = client.post(
        "/library/agents/agent-1/runs", json={"inputs": {"topic": "AI"}}
    )

    assert response.status_code == 202, response.text
    assert started.await_args.kwargs["inputs"] == {"topic": "AI"}


def test_a_422_does_not_hold_the_idempotency_key(
    client: fastapi.testclient.TestClient, started: mock.AsyncMock
) -> None:
    """Held, the key would answer the corrected retry with a 409 for 24 hours."""
    headers = {IDEMPOTENCY_HEADER: "retry-me"}
    path = "/library/agents/agent-1/runs"

    refused = client.post(path, json={"inputs": {}}, headers=headers)
    retried = client.post(path, json={"inputs": {"topic": "AI"}}, headers=headers)

    assert refused.status_code == 422, refused.text
    assert retried.status_code == 202, retried.text
    started.assert_awaited_once()


@pytest.fixture
def client(mocker: pytest_mock.MockFixture) -> fastapi.testclient.TestClient:
    store: dict[str, str] = {}

    async def set_(name: str, value: str, nx: bool = False, ex: int = 0) -> bool:
        if nx and name in store:
            return False
        store[name] = value
        return True

    async def delete(name: str) -> None:
        store.pop(name, None)

    mocker.patch(
        "backend.api.external.v2.idempotency.get_redis_async",
        new_callable=mock.AsyncMock,
        return_value=mock.Mock(
            set=mock.AsyncMock(side_effect=set_),
            get=mock.AsyncMock(side_effect=store.get),
            delete=mock.AsyncMock(side_effect=delete),
        ),
    )
    mocker.patch(
        "backend.api.external.v2.library.agents.graph_exec_limiter.check",
        new_callable=mock.AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.api.external.v2.library.helpers.get_credit_model",
        new_callable=mock.AsyncMock,
        return_value=mock.Mock(get_credits=mock.AsyncMock(return_value=100)),
    )
    mocker.patch(
        "backend.api.features.library.db.get_library_agent",
        new_callable=mock.AsyncMock,
        return_value=mock.Mock(
            organization_id=ORG_ID,
            graph_id="graph-1",
            graph_version=1,
            input_schema=INPUT_SCHEMA,
        ),
    )
    mocker.patch(
        "backend.data.graph.get_graph",
        new_callable=mock.AsyncMock,
        return_value=mock.Mock(version=1, input_schema=INPUT_SCHEMA),
    )

    app = fastapi.FastAPI()
    app.include_router(library_router, prefix="/library")
    app.include_router(schedules_router, prefix="/schedules")
    app.dependency_overrides[require_auth] = lambda: TenantContext(
        user_id="user-1",
        scopes=list(APIKeyPermission),
        type="api_key",
        organization_id=ORG_ID,
    )
    add_v2_exception_handlers(app)
    return fastapi.testclient.TestClient(app)


@pytest.fixture
def started(mocker: pytest_mock.MockFixture) -> mock.AsyncMock:
    """Every way a request can start work: a run now, or a schedule of them."""
    run = mocker.patch(
        "backend.executor.utils.add_graph_execution",
        new_callable=mock.AsyncMock,
        return_value=_run(),
    )
    scheduler = mocker.patch(
        "backend.api.external.v2.schedules.get_scheduler_client"
    ).return_value
    scheduler.add_execution_schedule = run
    return run


def _body(path: str, inputs: dict[str, Any]) -> dict[str, Any]:
    if path == "/schedules":
        return {
            "graph_id": "graph-1",
            "name": "Daily",
            "cron": "0 9 * * *",
            "timezone": "UTC",
            "inputs": inputs,
        }
    return {"inputs": inputs}


def _run() -> GraphExecutionMeta:
    return GraphExecutionMeta.model_construct(
        id="run-1",
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        preset_id=None,
        status=ExecutionStatus.QUEUED,
        started_at=None,
        ended_at=None,
        inputs={},
        credential_inputs=None,
        is_shared=False,
        share_token=None,
        stats=None,
        organization_id=ORG_ID,
        team_id=None,
    )
