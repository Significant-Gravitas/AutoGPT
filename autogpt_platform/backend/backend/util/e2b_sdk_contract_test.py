"""The locked e2b SDKs still work together and send the sandbox we ask for.

Drives the real ``e2b`` and ``e2b-code-interpreter`` packages against a fake
E2B API: an httpx ``MockTransport`` behind the SDK's own transport factory,
which both the control-plane client and the code interpreter's client use.
A lock where the two packages no longer fit (e2b 2.56 dropped the ``http2``
argument that code-interpreter 2.10.0 still passed, which broke every
``run_code``) or a change in the create request fails here, not in production.
No network, database or Redis.
"""

import json

import e2b.api.client_async as e2b_client_async
import httpx
import pytest
from e2b_code_interpreter import AsyncSandbox as CodeInterpreterSandbox

from backend.util import e2b_network
from backend.util.e2b_network import EgressOwner

_API_KEY = "e2b_" + "0" * 40
_OWNER = EgressOwner(kind="block", id="graph-exec-1", user_id="user-1")
_SANDBOX = {
    "templateID": "base",
    "sandboxID": "sbx",
    "clientID": "client",
    "envdVersion": "0.4.0",
    "envdAccessToken": "envd-token",
    "domain": "e2b.app",
}


@pytest.fixture
def fake_e2b(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path.rstrip("/")
        if request.method == "POST" and path.endswith("/sandboxes"):
            return httpx.Response(201, json=_SANDBOX)
        if request.method == "POST" and path == "/execute":
            lines = [
                {"type": "stdout", "text": "1\n", "timestamp": "2026-10-24T00:00:00Z"},
                {"type": "number_of_executions", "execution_count": 1},
            ]
            return httpx.Response(
                200, text="".join(json.dumps(line) + "\n" for line in lines)
            )
        return httpx.Response(404, json={"message": f"no fake for {path}"})

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(
        e2b_client_async, "get_httpx_transport", lambda *args, **kwargs: transport
    )
    monkeypatch.setattr(e2b_network, "proxy_address", lambda: None)
    return seen


def _create_body(seen: list[httpx.Request]) -> dict:
    creates = [r for r in seen if r.url.path.rstrip("/").endswith("/sandboxes")]
    assert len(creates) == 1, [r.url.path for r in seen]
    return json.loads(creates[0].content)


@pytest.mark.asyncio
async def test_code_interpreter_runs_code_against_the_locked_e2b(
    fake_e2b: list[httpx.Request],
):
    sandbox = await e2b_network.create_sandbox(
        CodeInterpreterSandbox,
        _OWNER,
        template="base",
        api_key=_API_KEY,
        timeout=600,
    )

    execution = await sandbox.run_code("print(1)")

    assert execution.logs.stdout == ["1\n"]
    execute = next(r for r in fake_e2b if r.url.path == "/execute")
    assert execute.headers["X-Access-Token"] == "envd-token"


@pytest.mark.asyncio
async def test_create_asks_for_the_timeout_and_internet_access_we_rely_on(
    fake_e2b: list[httpx.Request],
):
    await e2b_network.create_sandbox(
        CodeInterpreterSandbox,
        _OWNER,
        template="base",
        api_key=_API_KEY,
        timeout=600,
    )

    body = _create_body(fake_e2b)
    assert body["timeout"] == 600
    assert body["allow_internet_access"] is True
    # Sandboxes have always been created secured; newer SDKs drop the field
    # because the API no longer offers an insecure mode.
    assert body.get("secure", True) is True
