"""Schedule routes while the scheduler is restarting or hung.

The real route code and the real scheduler RPC client (httpx), pointed at a
scheduler that is either down (nothing listening, as during a deploy) or hung
(it accepts the connection and never answers). Before this, the first case
answered 500 "All connection attempts failed" (AUTOGPT-SERVER-A6R/A6S) and the
second waited out the 300s RPC timeout until the proxy gave up with a 504.
"""

import inspect
import socket
import time
import uuid
from typing import AsyncIterator, Iterator, Literal
from unittest.mock import Mock

import httpx
import pytest
from autogpt_libs.auth.models import RequestContext
from fastapi import FastAPI

from backend.api.features import schedule_availability
from backend.api.features.admin import diagnostics_admin_routes
from backend.api.features.schedules import routes
from backend.api.model import CreateGraph
from backend.data import diagnostics
from backend.data.onboarding import OnboardingStep, get_user_onboarding
from backend.data.user import get_or_create_user
from backend.executor.scheduler import Scheduler
from backend.usecases.sample import create_test_graph
from backend.util import clients
from backend.util.test import SpinTestServer

_REQUEST_TIMEOUT_S = 1.0
# Far below the 30s proxy cut-off and the 300s RPC default, generous for CI.
_PROMPT_S = 4.0

Outage = Literal["down", "hung"]


@pytest.fixture
def scheduler_port(request: pytest.FixtureRequest) -> Iterator[int]:
    outage: Outage = request.param
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    if outage == "down":
        sock.close()
        yield port
        return
    # Listening but never accepting: the kernel completes the handshake and
    # buffers the request, and no answer ever comes.
    sock.listen(64)
    try:
        yield port
    finally:
        sock.close()


@pytest.fixture
def scheduler_client(
    scheduler_port: int, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    monkeypatch.setattr(Scheduler, "get_host", classmethod(lambda cls: "127.0.0.1"))
    monkeypatch.setattr(Scheduler, "get_port", classmethod(lambda cls: scheduler_port))
    for module in (clients, schedule_availability):
        config = module.settings.config
        monkeypatch.setattr(
            config, "scheduler_request_timeout_seconds", _REQUEST_TIMEOUT_S
        )
        # A hung write waits out the RPC's own timeout (300s in production);
        # shortened so the test reaches the scheduler_unconfirmed answer.
        monkeypatch.setattr(config, "rpc_client_call_timeout", 1)
        monkeypatch.setattr(config, "scheduler_connect_timeout_seconds", 0.5)
    # The real factory, uncached, so each test gets a client for its own port.
    real_factory = inspect.unwrap(clients.get_interactive_scheduler_client)
    monkeypatch.setattr(routes, "get_interactive_scheduler_client", real_factory)
    monkeypatch.setattr(diagnostics, "get_interactive_scheduler_client", real_factory)
    yield


def _app(user_id: str) -> FastAPI:
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(diagnostics_admin_routes.router)
    app.dependency_overrides[routes.requires_user] = lambda: None
    app.dependency_overrides[diagnostics_admin_routes.requires_admin_user] = (
        lambda: Mock(user_id=user_id)
    )
    app.dependency_overrides[routes.get_user_id] = lambda: user_id
    app.dependency_overrides[routes.get_request_context] = lambda: RequestContext(
        user_id=user_id,
        org_id=None,
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )
    return app


async def _http(user_id: str) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_app(user_id))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def http(scheduler_client: None) -> AsyncIterator[httpx.AsyncClient]:
    async for c in _http("user-1"):
        yield c


async def _timed(call) -> tuple[httpx.Response, float]:
    started = time.monotonic()
    response = await call
    return response, time.monotonic() - started


def _assert_503(response: httpx.Response, code: str) -> None:
    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == code
    assert response.headers["Retry-After"]


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduler_port", ["down", "hung"], indirect=True)
@pytest.mark.parametrize(
    "path",
    [
        "/schedules",
        "/schedules/followups",
        "/graphs/graph-1/schedules",
        "/admin/diagnostics/schedules",
        "/admin/diagnostics/schedules/all",
        "/admin/diagnostics/schedules/orphaned",
    ],
)
async def test_schedule_reads_answer_503_promptly(http: httpx.AsyncClient, path: str):
    response, elapsed = await _timed(http.get(path))

    _assert_503(response, "scheduler_unavailable")
    assert elapsed < _PROMPT_S


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scheduler_port", "code"),
    [("down", "scheduler_unavailable"), ("hung", "scheduler_unconfirmed")],
    indirect=["scheduler_port"],
)
async def test_schedule_delete_answers_503_promptly(http: httpx.AsyncClient, code: str):
    response, elapsed = await _timed(http.delete("/schedules/schedule-1"))

    _assert_503(response, code)
    assert elapsed < _PROMPT_S


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scheduler_port", "code"),
    [("down", "scheduler_unavailable"), ("hung", "scheduler_unconfirmed")],
    indirect=["scheduler_port"],
)
async def test_schedule_edit_answers_503_promptly(http: httpx.AsyncClient, code: str):
    response, elapsed = await _timed(
        http.patch("/schedules/schedule-1", json={"name": "Renamed"})
    )

    _assert_503(response, code)
    assert elapsed < _PROMPT_S


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduler_port", ["down", "hung"], indirect=True)
async def test_admin_orphan_cleanup_answers_503_promptly(http: httpx.AsyncClient):
    # Cleanup reads the schedule list before deleting anything; with the
    # scheduler out, that read fails and nothing is deleted.
    response, elapsed = await _timed(
        http.post(
            "/admin/diagnostics/schedules/cleanup-orphaned",
            json={"schedule_ids": ["schedule-1"]},
        )
    )

    _assert_503(response, "scheduler_unavailable")
    assert elapsed < _PROMPT_S


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("scheduler_port", "code"),
    [("down", "scheduler_unavailable"), ("hung", "scheduler_unconfirmed")],
    indirect=["scheduler_port"],
)
async def test_schedule_create_answers_503_and_changes_nothing(
    server: SpinTestServer, scheduler_client: None, code: str
):
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"scheduler-outage-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Scheduler Outage",
        }
    )
    graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=create_test_graph()), user.id
    )

    async for http in _http(user.id):
        response, elapsed = await _timed(
            http.post(
                f"/graphs/{graph.id}/schedules",
                json={
                    "name": "Every morning",
                    "cron": "0 9 * * *",
                    "inputs": {},
                    "graph_version": graph.version,
                },
            )
        )

    _assert_503(response, code)
    assert elapsed < _PROMPT_S
    # The route stops at the scheduler: the follow-on onboarding write, which
    # only belongs to a schedule that exists, never happens.
    onboarding = await get_user_onboarding(user.id)
    assert OnboardingStep.SCHEDULE_AGENT not in onboarding.completedSteps
