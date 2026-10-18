import pytest
from aiohttp import web

from backend.copilot.sdk.model_gateway.front import ModelGateway
from backend.copilot.sdk.model_gateway.testing.cli_version import (
    assert_cli_version,
    binary_version,
)
from backend.util import encryption


@pytest.fixture(autouse=True)
def fake_seal_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        encryption, "ENCRYPTION_KEY", "gateway-test-only-encryption-key"
    )


@pytest.fixture(autouse=True)
def audit_real_cli_version(request, monkeypatch):
    if "cli" not in request.fixturenames:
        yield
        return
    expected = binary_version(request.getfixturevalue("cli"))
    seen = []
    messages = ModelGateway._messages

    async def observe_cli_header(
        gateway: ModelGateway, incoming: web.Request
    ) -> web.StreamResponse:
        seen.append(incoming.headers.get("User-Agent", ""))
        return await messages(gateway, incoming)

    monkeypatch.setattr(ModelGateway, "_messages", observe_cli_header)
    yield
    request.node.user_properties.extend(
        [("cli_binary_version", expected), ("cli_user_agents", sorted(set(seen)))]
    )
    assert_cli_version(seen, expected)
