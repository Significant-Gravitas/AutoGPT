import hashlib
import json
import logging
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from redis.asyncio.lock import Lock

from backend.copilot.sdk.model_gateway.adapters.anthropic_passthrough import (
    AnthropicPassthrough,
)
from backend.copilot.sdk.model_gateway.adapters.codex_input import responses_payload
from backend.copilot.sdk.model_gateway.adapters.codex_responses import CodexResponses
from backend.copilot.sdk.model_gateway.adapters.codex_signature import (
    reasoning_tag,
    seal,
    unseal,
)
from backend.copilot.sdk.model_gateway.adapters.openai_request import chat_payload
from backend.copilot.sdk.model_gateway.errors import GatewayError
from backend.copilot.sdk.model_gateway.front import ModelGateway
from backend.copilot.sdk.model_gateway.normalize import normalize_request
from backend.copilot.sdk.model_gateway.route import (
    ApiKeyCredential,
    CodexLeaseCredential,
    GatewayRoute,
)
from backend.copilot.sdk.model_gateway.testing.codex_lease import (
    codex_route,
    fake_backend_client,
    fake_codex_lease,
)
from backend.copilot.sdk.model_gateway.testing.regressions import BODY, post, terminal
from backend.integrations.credential_lease import CredentialLease
from backend.util import encryption


async def test_passthrough_preserves_tool_input_data_that_looks_like_an_image():
    domain_data = {
        "type": "image",
        "source": {"type": "url", "url": "https://domain.test"},
    }
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call",
                        "name": "read",
                        "input": domain_data,
                    }
                ],
            }
        ],
        "metadata": domain_data,
    }
    route = GatewayRoute(
        adapter="anthropic_passthrough", model="m", base_url="http://127.0.0.1:9"
    )
    adapter = AnthropicPassthrough(route)
    try:
        payload = adapter._payload(body, stream=True)
    finally:
        await adapter.aclose()
    assert payload["messages"] == body["messages"]
    assert payload["metadata"] == body["metadata"]


@pytest.mark.parametrize(
    "mutation", ["public-tag", "ciphertext", "mac", "truncated", "key-rotation"]
)
def test_reasoning_requires_authenticity_and_the_current_platform_key(
    monkeypatch, mutation
):
    tag = reasoning_tag("acct-target", "model")
    signature = seal("encrypted.original", tag)
    if mutation == "public-tag":
        public = hashlib.sha256(b"acct-target\nmodel").hexdigest()[:16]
        signature = f"agptcx1.{public}.attacker-chosen"
    elif mutation == "ciphertext":
        signature = signature.replace("original", "attacker-chosen")
    elif mutation == "mac":
        prefix, middle, content = signature.split(".", 2)
        signature = f"{prefix}.{middle[:-1]}0.{content}"
    elif mutation == "truncated":
        signature = signature[:-1]
    else:
        monkeypatch.setattr(encryption, "ENCRYPTION_KEY", "rotated-test-key")
    assert unseal(signature, tag) is None


@pytest.mark.parametrize(
    "adapter", ["anthropic_passthrough", "openai_chat", "codex_responses"]
)
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://10.0.0.1/x",
        "http://169.254.169.254/x",
        "file:///secret",
        "http://public.test/redirect-to-private",
        "http://rebind.test/x",
    ],
)
async def test_image_urls_are_replaced_in_place_including_tool_results(adapter, url):
    image = {"type": "image", "source": {"type": "url", "url": url}}
    inline = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
    }
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    image,
                    inline,
                    {"type": "text", "text": "after"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "call",
                        "content": [image, inline],
                    },
                ],
            }
        ],
        "metadata": {"keep": "untouched"},
    }
    if adapter == "codex_responses":
        payload, _ = responses_payload(
            normalize_request(body), codex_route("m"), reasoning_tag("acct-test", "m")
        )
    else:
        route = GatewayRoute(
            adapter=adapter, model="m", base_url="http://127.0.0.1:9/v1"
        )
        if adapter == "openai_chat":
            payload, _ = chat_payload(normalize_request(body), route)
        else:
            implementation = AnthropicPassthrough(route)
            try:
                payload = implementation._payload(body, stream=True)
            finally:
                await implementation.aclose()
            assert payload["metadata"] == body["metadata"]
            assert payload["messages"][0]["content"][1]["type"] == "text"
            assert payload["messages"][0]["content"][2] == inline
    encoded = json.dumps(payload)
    assert url not in encoded
    assert "image omitted" in encoded
    assert "aGVsbG8=" in encoded
    assert body["messages"][0]["content"][1] == image


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_codex_redirects_are_not_followed(status):
    seen = []

    def handle(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(
                status, headers={"location": "https://redirect-target.test/collect"}
            )
        return httpx.Response(200, content="data: " + json.dumps(terminal()) + "\n\n")

    async with fake_backend_client(
        "https://chatgpt.com/backend-api/codex", transport=httpx.MockTransport(handle)
    ) as client:
        adapter = CodexResponses(codex_route("m"), client=client)
        with pytest.raises(GatewayError) as raised:
            await adapter.complete(BODY)
    assert raised.value.status == 502
    assert len(seen) == 1


async def test_codex_debug_logs_never_include_content_or_credentials(
    monkeypatch, caplog
):
    markers = [
        "private-prompt-987654321",
        "private-reasoning-987654321",
        "acct-sensitive",
    ]
    lease = fake_codex_lease(account_id=markers[-1])
    markers.append(lease.credentials.access_token.get_secret_value())
    route = codex_route("m", lease=lease)
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "",
                        "signature": seal(markers[1], reasoning_tag(markers[-2], "m")),
                    }
                ],
            },
            {"role": "user", "content": markers[0]},
        ]
    }
    monkeypatch.setenv("OPENAI_LOG", "debug")
    caplog.set_level(logging.DEBUG)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content="data: " + json.dumps(terminal()) + "\n\n"
        )
    )
    async with fake_backend_client(
        "https://chatgpt.com/backend-api/codex", transport=transport
    ) as client:
        await CodexResponses(route, client=client).complete(body)
    assert all(marker not in caplog.text for marker in markers)


async def test_unknown_error_fields_do_not_leak_to_failure_or_front(caplog):
    route = codex_route("m")
    token = route.secret_values()[0]
    caplog.set_level(logging.DEBUG)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, json={"unexpected": token})
    )
    async with fake_backend_client(
        "https://chatgpt.com/backend-api/codex", transport=transport
    ) as client:
        adapter = CodexResponses(route, client=client)
        with pytest.raises(GatewayError) as raised:
            await adapter.complete(BODY)
        async with ModelGateway(route, upstream=adapter) as gateway:
            status, raw = await post(gateway, streaming=False)
    assert status == 401 and token not in raw and token not in caplog.text
    assert token not in str(raised.value)
    assert raised.value.failure is not None
    assert token not in raised.value.failure.model_dump_json()
    assert gateway.last_failure is not None
    assert token not in gateway.last_failure.model_dump_json()


@pytest.mark.parametrize("key", ["a", "ab", "abc"])
def test_routes_reject_short_operator_keys(key):
    with pytest.raises(ValidationError, match="four"):
        GatewayRoute(
            adapter="openai_chat",
            model="m",
            base_url="http://127.0.0.1:9",
            credential=ApiKeyCredential(api_key=SecretStr(key)),
        )


async def test_checkpointed_lease_refreshes_auth_and_keeps_reasoning():
    old = fake_codex_lease().credentials
    lock = AsyncMock(spec=Lock)
    lock.locked.return_value = lock.owned.return_value = True
    lease = CredentialLease(old, lock, AsyncMock())
    route = GatewayRoute(
        adapter="codex_responses",
        model="m",
        credential=CodexLeaseCredential(lease=lease),
    )
    fresh = old.model_copy(update={"access_token": SecretStr("fresh-test-token")})
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, content="data: " + json.dumps(terminal()) + "\n\n")

    async with fake_backend_client(
        "https://chatgpt.com/backend-api/codex", transport=httpx.MockTransport(handle)
    ) as client:
        adapter = CodexResponses(route, client=client)
        signature = seal("encrypted", reasoning_tag("acct-test", "m"))
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "signature": signature}],
                },
                {"role": "user", "content": "go"},
            ]
        }
        await adapter.complete(body)
        await lease.checkpoint(fresh)
        await adapter.complete(body)
    assert [request.headers["authorization"] for request in seen] == [
        "Bearer " + old.access_token.get_secret_value(),
        "Bearer fresh-test-token",
    ]
    assert all(
        json.loads(request.content)["input"][0]["encrypted_content"] == "encrypted"
        for request in seen
    )
