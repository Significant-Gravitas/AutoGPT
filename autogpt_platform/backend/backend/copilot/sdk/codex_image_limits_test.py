import pytest
from aiohttp import ClientSession

from backend.copilot.pending_messages import MAX_PENDING_MESSAGES
from backend.copilot.sdk import codex_compat_gateway
from backend.copilot.sdk.codex_compat_gateway import CodexAnthropicGateway
from backend.copilot.sdk.codex_compat_gateway_test import (
    _FakeAgentSession,
    _FakeTransport,
    _headers,
    _lease,
    _tool_request,
    _tool_use_id,
)
from backend.copilot.sdk.service import _is_prompt_too_long
from backend.copilot.tool_image import MAX_IMAGE_BYTES, ToolImage


@pytest.mark.parametrize("count", [4, MAX_PENDING_MESSAGES])
async def test_max_size_images_survive_gateway_count_start_continue_and_retry(count):
    image = ToolImage(
        mime_type="image/png", data_base64="A" * (4 * ((MAX_IMAGE_BYTES + 2) // 3))
    )
    blocks = [image.anthropic()] * count
    session = _FakeAgentSession(use_tool=True)
    async with CodexAnthropicGateway(
        transport=_FakeTransport(session), credential_lease=_lease(), model="gpt-5.4"
    ) as gateway, ClientSession() as client:
        payload = _tool_request("Inspect these pages")
        payload["messages"] = [{"role": "user", "content": blocks}]
        count_response = await client.post(
            f"{gateway.base_url}/v1/messages/count_tokens",
            headers=_headers(gateway),
            json=payload,
        )
        await count_response.read()
        assert count_response.status == 200
        started = await client.post(
            f"{gateway.base_url}/v1/messages",
            headers=_headers(gateway),
            json=payload,
        )
        assert started.status == 200
        call_id = _tool_use_id(await started.json())
        continuation = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": blocks,
                        }
                    ],
                }
            ]
        }
        responses = []
        for _ in range(2):
            response = await client.post(
                f"{gateway.base_url}/v1/messages",
                headers=_headers(gateway),
                json=continuation,
            )
            body = await response.read()
            assert response.status == 200
            responses.append(body)
    expected = [image.openai()["image_url"]["url"]] * count
    assert len(session.requests) == 1
    assert session.requests[0].image_urls == expected
    assert session.tool_result is not None
    assert session.tool_result.image_urls == expected
    assert responses[0] == responses[1]


@pytest.mark.parametrize("endpoint", ["messages", "messages/count_tokens"])
@pytest.mark.parametrize("image_history", [False, True])
async def test_gateway_size_error_explains_how_to_recover(
    monkeypatch, endpoint, image_history
):
    monkeypatch.setattr(codex_compat_gateway, "_MAX_REQUEST_BYTES", 512)
    image = ToolImage(mime_type="image/png", data_base64="A" * 256)
    messages = (
        [{"role": "user", "content": [image.anthropic()]}] * 3
        if image_history
        else [{"role": "user", "content": "x" * 1024}]
    )
    session = _FakeAgentSession()
    async with CodexAnthropicGateway(
        transport=_FakeTransport(session), credential_lease=_lease(), model="gpt-5.4"
    ) as gateway, ClientSession() as client:
        response = await client.post(
            f"{gateway.base_url}/v1/{endpoint}",
            headers=_headers(gateway),
            json={"messages": messages},
        )
        assert response.status == 413
        body = await response.json()
    assert body["type"] == "error"
    assert "fewer or smaller images" in body["error"]["message"]
    assert _is_prompt_too_long(RuntimeError(body["error"]["message"]))
    assert not session.requests
