"""The `/mcp` endpoint answers over HTTP once the host app's lifespan has run.

Starlette never runs a mounted app's lifespan, and FastMCP starts the session
manager that every request goes through in its lifespan. Mounted as-is, every
request to `/external-api/v2/mcp/` was a 500: "Task group is not initialized".
"""

import contextlib
from unittest.mock import AsyncMock

import fastapi
import pytest_mock
from fastapi.testclient import TestClient

from backend.api.external.v2.mcp_server import MCPMount, TenantedAccessToken

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}
HEADERS = {
    "Authorization": "Bearer agpt_test",
    "Accept": "application/json, text/event-stream",
}


def _host(mount: MCPMount) -> fastapi.FastAPI:
    """A host app wired the way the platform's own app wires the mount."""

    @contextlib.asynccontextmanager
    async def lifespan(_: fastapi.FastAPI):
        async with mount.running():
            yield

    host = fastapi.FastAPI(lifespan=lifespan)
    host.mount("/mcp", mount)
    return host


def test_an_authenticated_initialize_succeeds_through_the_mount(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "backend.api.external.v2.mcp_server.ExternalAPITokenVerifier.verify_token",
        new_callable=AsyncMock,
        return_value=TenantedAccessToken(
            token="agpt_test",
            client_id="user-1",
            scopes=["IDENTITY"],
            organization_id="org-1",
        ),
    )

    # A public hostname: FastMCP's default host check admitted only localhost.
    with TestClient(_host(MCPMount()), base_url="https://backend.agpt.co") as client:
        response = client.post("/mcp/", headers=HEADERS, json=INITIALIZE)

    assert response.status_code == 200, response.text
    assert "autogpt-platform" in response.text


def test_the_host_lifespan_can_run_more_than_once(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A session manager runs once, so each lifespan needs its own server."""
    mount = MCPMount()
    for _ in range(2):
        with TestClient(_host(mount)):
            pass


def test_a_server_that_fails_to_start_leaves_the_host_up_answering_503(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The REST API must not go down with its MCP endpoint."""
    mocker.patch(
        "backend.api.external.v2.mcp_server.create_mcp_app",
        side_effect=RuntimeError("no tools today"),
    )

    with TestClient(_host(MCPMount())) as client:
        response = client.post("/mcp/", headers=HEADERS, json=INITIALIZE)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"


def test_before_the_lifespan_runs_the_mount_answers_in_the_v2_envelope() -> None:
    host = fastapi.FastAPI()
    host.mount("/mcp", MCPMount())

    response = TestClient(host).post("/mcp/", headers=HEADERS, json=INITIALIZE)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"
