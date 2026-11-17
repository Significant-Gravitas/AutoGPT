"""Schedule reads answer a fast 503 while the scheduler is out.

The real routes, mounted as ``rest_api`` mounts them, the real exception
handlers and the real scheduler RPC client (httpx over TCP), pointed at a
scheduler that is down (nothing listening: a deploy's gap, where kube-proxy
refuses the connection) or hung (it takes the connection and never answers).
Unbounded, a down scheduler surfaces as a 500 and a hung one holds the request
for the 300s RPC timeout, long past the 30s frontend proxy.
"""

import asyncio
import inspect
import logging
import socket
from typing import AsyncIterator, Iterator
from unittest.mock import AsyncMock

import httpx
import pytest
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from fastapi import FastAPI

from backend.api.features.admin import diagnostics_admin_routes
from backend.api.features.schedules import routes
from backend.api.utils import exceptions as api_exceptions
from backend.api.utils.exceptions import (
    SCHEDULER_RETRY_AFTER_SECONDS,
    SCHEDULER_UNAVAILABLE_DETAIL,
    add_exception_handlers,
)
from backend.data import diagnostics
from backend.executor.scheduler import Scheduler, SchedulerClient
from backend.util import clients
from backend.util.service import HTTPServerError
from backend.util.settings import Config

# The cap under test; production's is ``schedule_list_timeout_seconds``.
_CAP_S = 0.5
# Far over the cap, far under the 300s a read waits without one: a regression
# fails here instead of hanging the run.
_ANSWERED_WITHIN_S = 20.0

SCHEDULE_READS = [
    ("GET", "/api/schedules", None),
    ("GET", "/api/schedules/followups", None),
    ("GET", "/api/graphs/graph-1/schedules", None),
    ("GET", "/api/admin/diagnostics/schedules", None),
    ("GET", "/api/admin/diagnostics/schedules/all", None),
    ("GET", "/api/admin/diagnostics/schedules/orphaned", None),
    (
        "POST",
        "/api/admin/diagnostics/schedules/cleanup-orphaned",
        {"schedule_ids": ["schedule-1"]},
    ),
]


@pytest.fixture(params=["down", "hung"])
def scheduler_port(request: pytest.FixtureRequest) -> Iterator[int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    if request.param == "down":
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
async def scheduler_out(
    scheduler_port: int, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[SchedulerClient]:
    """The real scheduler client, built by the real factory, aimed at the outage."""
    monkeypatch.setattr(Scheduler, "get_host", classmethod(lambda cls: "127.0.0.1"))
    monkeypatch.setattr(Scheduler, "get_port", classmethod(lambda cls: scheduler_port))
    monkeypatch.setattr(
        clients.settings.config, "schedule_list_timeout_seconds", _CAP_S
    )
    # Each test starts, and leaves, with no outage under way.
    monkeypatch.setattr(clients, "_scheduler_out_since", None)
    # Unwrapped from its thread cache, so no other test gets this client.
    client = inspect.unwrap(clients.get_scheduler_client)()
    for module in (routes, diagnostics):
        monkeypatch.setattr(module, "get_scheduler_client", lambda: client)
    yield client
    await client.aclose()


@pytest.fixture
async def http(scheduler_out: SchedulerClient) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    app.include_router(diagnostics_admin_routes.router, prefix="/api")
    add_exception_handlers(app)
    app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": "admin-1",
        "role": "admin",
        "email": "admin@example.com",
    }
    # No org, so no team lookup reaches the database before the scheduler.
    app.dependency_overrides[get_request_context] = lambda: RequestContext(
        user_id="admin-1",
        org_id=None,
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )
    # A 500 comes back as a response rather than raising into the test.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.parametrize(("method", "path", "body"), SCHEDULE_READS)
async def test_a_schedule_read_answers_503_promptly_while_the_scheduler_is_out(
    http: httpx.AsyncClient, method: str, path: str, body: dict | None
):
    response = await asyncio.wait_for(
        http.request(method, path, json=body), timeout=_ANSWERED_WITHIN_S
    )

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == SCHEDULER_UNAVAILABLE_DETAIL
    assert response.headers["Retry-After"] == str(SCHEDULER_RETRY_AFTER_SECONDS)


async def test_an_outage_deletes_no_orphaned_schedule(
    http: httpx.AsyncClient,
    scheduler_out: SchedulerClient,
    monkeypatch: pytest.MonkeyPatch,
):
    # The cleanup reads the schedules first, and that read fails.
    deletes = AsyncMock()
    monkeypatch.setattr(scheduler_out, "delete_schedule", deletes)

    response = await asyncio.wait_for(
        http.post(
            "/api/admin/diagnostics/schedules/cleanup-orphaned",
            json={"schedule_ids": ["schedule-1"]},
        ),
        timeout=_ANSWERED_WITHIN_S,
    )

    assert response.status_code == 503, response.text
    deletes.assert_not_awaited()


@pytest.mark.parametrize("scheduler_port", ["down"], indirect=True)
async def test_only_an_outage_longer_than_any_deploy_is_logged_as_an_error(
    http: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    # Sentry files errors, not warnings: a deploy's gap must not page anyone,
    # and a scheduler that stays down must.
    now = [1000.0]
    monkeypatch.setattr(clients, "monotonic", lambda: now[0])

    async def logged_levels() -> list[str]:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=api_exceptions.__name__):
            await asyncio.wait_for(
                http.get("/api/schedules"), timeout=_ANSWERED_WITHIN_S
            )
        return [
            r.levelname for r in caplog.records if r.name == api_exceptions.__name__
        ]

    assert await logged_levels() == ["WARNING"]
    now[0] += clients.SCHEDULER_SUSTAINED_OUTAGE_SECONDS
    assert await logged_levels() == ["ERROR"]

    async def scheduler_back() -> list:
        return []

    await clients.read_schedules(scheduler_back())
    now[0] += clients.SCHEDULER_SUSTAINED_OUTAGE_SECONDS
    # A new outage starts its own clock.
    assert await logged_levels() == ["WARNING"]


async def test_a_scheduler_that_answers_with_an_error_is_not_an_outage():
    # Its own failure stays a 500 that someone investigates, not a calm 503.
    async def scheduler_failed() -> None:
        raise HTTPServerError(500, "the jobstore query failed")

    with pytest.raises(HTTPServerError):
        await clients.read_schedules(scheduler_failed())


def test_the_read_cap_answers_before_the_frontend_proxy_gives_up():
    # The proxy cuts a request off at 30s; past it, the user gets the proxy's
    # error and the backend keeps waiting on a request nobody will read.
    assert Config.model_fields["schedule_list_timeout_seconds"].default < 30
