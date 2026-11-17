"""Schedule reads while the scheduler is out.

The real routes, mounted as ``rest_api`` mounts them, the real exception
handlers and the real scheduler RPC client (httpx over TCP), pointed at a
scheduler that is down (nothing listening: a deploy's gap, where kube-proxy
refuses the connection) or hung (it takes the connection and never answers).
Before, a down scheduler answered 500 "All connection attempts failed"
(AUTOGPT-SERVER-A6R/A6S) and a hung one held the request for the 300s RPC
timeout, long after the frontend proxy gave up at 30s.
"""

import asyncio
import inspect
import socket
from typing import AsyncIterator, Iterator

import httpx
import pytest
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from fastapi import FastAPI

from backend.api.features.admin import diagnostics_admin_routes
from backend.api.features.schedules import routes
from backend.api.utils.exceptions import (
    SCHEDULER_RETRY_AFTER_SECONDS,
    SCHEDULER_UNAVAILABLE_DETAIL,
    add_exception_handlers,
)
from backend.data import diagnostics
from backend.executor.scheduler import Scheduler
from backend.util import clients
from backend.util.service import HTTPServerError

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
    # Reads the schedules before deleting any, so nothing is deleted.
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
) -> AsyncIterator[None]:
    """The real scheduler client, built by the real factory, aimed at the outage."""
    monkeypatch.setattr(Scheduler, "get_host", classmethod(lambda cls: "127.0.0.1"))
    monkeypatch.setattr(Scheduler, "get_port", classmethod(lambda cls: scheduler_port))
    monkeypatch.setattr(
        clients.settings.config, "schedule_list_timeout_seconds", _CAP_S
    )
    # Unwrapped from its thread cache, so no other test gets this client.
    client = inspect.unwrap(clients.get_scheduler_client)()
    for module in (routes, diagnostics):
        monkeypatch.setattr(module, "get_scheduler_client", lambda: client)
    yield
    await client.aclose()


@pytest.fixture
async def http(scheduler_out: None) -> AsyncIterator[httpx.AsyncClient]:
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


async def test_a_scheduler_that_answers_with_an_error_is_not_an_outage():
    # Its own failure stays a 500 that someone investigates, not a calm 503.
    async def scheduler_failed() -> None:
        raise HTTPServerError(500, "the jobstore query failed")

    with pytest.raises(HTTPServerError):
        await clients.read_schedules(scheduler_failed())


def test_the_read_cap_answers_before_the_frontend_proxy_gives_up():
    # The proxy cuts a request off at 30s; past it, the user gets the proxy's
    # error and the backend keeps waiting on a request nobody will read.
    assert clients.settings.config.schedule_list_timeout_seconds < 30
