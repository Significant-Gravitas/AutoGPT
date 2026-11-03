import pytest
from aiohttp import ClientSession

from backend.copilot.pending_messages import MAX_PENDING_MESSAGES
from backend.copilot.sdk.model_gateway import front
from backend.copilot.sdk.model_gateway.front import ModelGateway
from backend.copilot.sdk.model_gateway.front_test import _ROUTE, _FakeUpstream, _post
from backend.copilot.sdk.service import _is_prompt_too_long
from backend.copilot.tool_image import MAX_IMAGE_BYTES, ToolImage


@pytest.mark.parametrize("count", [4, MAX_PENDING_MESSAGES])
async def test_max_size_images_survive_count_request_and_retry(count):
    image = ToolImage(
        mime_type="image/png", data_base64="A" * (4 * ((MAX_IMAGE_BYTES + 2) // 3))
    )
    upstream = _FakeUpstream()
    body = {"messages": [{"role": "user", "content": [image.anthropic()] * count}]}
    route = _ROUTE.model_copy(update={"adapter": "codex_responses"})
    async with ModelGateway(route, upstream=upstream) as gateway:
        async with ClientSession() as client:
            async with client.post(
                f"{gateway.base_url}/v1/messages/count_tokens",
                json=body,
                headers={"x-api-key": gateway.auth_token},
            ) as response:
                assert response.status == 200
                assert (await response.json())["input_tokens"] >= count * 1000
        first = (await _post(gateway, body))[2]
        assert (await _post(gateway, body))[2] == first
    assert len(upstream.bodies) == 1
    assert upstream.bodies[0] == body


@pytest.mark.parametrize("endpoint", ["messages", "messages/count_tokens"])
@pytest.mark.parametrize("image_history", [False, True])
async def test_gateway_size_error_explains_how_to_recover(
    monkeypatch, endpoint, image_history
):
    monkeypatch.setattr(front, "_MAX_REQUEST_BYTES", 512)
    image = ToolImage(mime_type="image/png", data_base64="A" * 256)
    messages = (
        [{"role": "user", "content": [image.anthropic()]}] * 3
        if image_history
        else [{"role": "user", "content": "x" * 1024}]
    )
    upstream = _FakeUpstream()
    async with ModelGateway(
        _ROUTE, upstream=upstream
    ) as gateway, ClientSession() as client:
        async with client.post(
            f"{gateway.base_url}/v1/{endpoint}",
            json={"messages": messages},
            headers={"x-api-key": gateway.auth_token},
        ) as response:
            assert response.status == 413
            body = await response.json()
    assert "fewer or smaller images" in body["error"]["message"]
    assert _is_prompt_too_long(RuntimeError(body["error"]["message"]))
    assert not upstream.bodies
