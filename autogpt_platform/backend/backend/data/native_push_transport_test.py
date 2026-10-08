import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from backend.data import native_push_transport as transport
from backend.data.native_push_subscription import NativePushSubscriptionDTO


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "environment,host",
    [("sandbox", "api.sandbox.push.apple.com"), ("production", "api.push.apple.com")],
)
async def test_apns_uses_signed_http2_and_only_a_generic_alert(
    tmp_path, monkeypatch, environment, host
):
    key = ec.generate_private_key(ec.SECP256R1())
    path = tmp_path / "test-key.p8"
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    config = SimpleNamespace(
        apns_private_key_path=str(path),
        apns_key_id="test-key",
        apns_team_id="test-team",
    )
    monkeypatch.setattr(transport, "Settings", lambda: SimpleNamespace(config=config))
    requests = []
    client = httpx.AsyncClient
    monkeypatch.setattr(
        transport.httpx,
        "AsyncClient",
        lambda **kw: client(
            **kw,
            transport=httpx.MockTransport(
                lambda request: (requests.append(request), httpx.Response(200))[1]
            )
        ),
    )
    sub = NativePushSubscriptionDTO(
        id="binding",
        provider="apns",
        environment=environment,
        token="a" * 64,
        origin="https://platform.agpt.co",
    )
    await transport.send_apns(
        sub, "There's an update in your chat.", "/home?sessionId=chat"
    )
    request = requests[0]
    assert request.url.host == host
    assert request.headers["apns-topic"] == "com.agpt.mobile"
    bearer = request.headers["authorization"].removeprefix("bearer ")
    assert (
        jwt.decode(bearer, key.public_key(), algorithms=["ES256"])["iss"] == "test-team"
    )
    data = json.loads(request.content)
    assert data["binding_id"] == "binding"
    assert set(data) == {"aps", "binding_id", "origin", "path"}


@pytest.mark.asyncio
async def test_fcm_data_messages_allow_account_validation_before_display(
    tmp_path, monkeypatch
):
    path = tmp_path / "service-account.json"
    path.write_text("{}")
    monkeypatch.setattr(
        transport,
        "Settings",
        lambda: SimpleNamespace(
            config=SimpleNamespace(fcm_service_account_path=str(path))
        ),
    )
    requests = []
    client = httpx.AsyncClient
    monkeypatch.setattr(
        transport.httpx,
        "AsyncClient",
        lambda **kw: client(
            **kw,
            transport=httpx.MockTransport(
                lambda request: (requests.append(request), httpx.Response(200))[1]
            )
        ),
    )
    sub = NativePushSubscriptionDTO(
        id="binding",
        provider="fcm",
        environment="production",
        token="device-token",
        origin="https://platform.agpt.co",
    )
    with patch.object(
        transport,
        "_fcm_credentials",
        return_value=("test-project", "short-lived-test-token"),
    ):
        await transport.send_fcm(
            sub, "Your team needs a response.", "/mobile?tab=attention"
        )
    message = json.loads(requests[0].content)["message"]
    assert "notification" not in message
    assert message["data"]["binding_id"] == "binding"
    assert message["android"] == {"priority": "high", "ttl": "3600s"}
    assert requests[0].url.path == "/v1/projects/test-project/messages:send"
