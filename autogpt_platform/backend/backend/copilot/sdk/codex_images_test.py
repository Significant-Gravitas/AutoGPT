import io

import pytest
from PIL import Image

from backend.copilot.tool_image import prepare_image
from backend.integrations.codex.http_session_test import (
    TOOL,
    _request,
    _session,
    _text_turn,
    _tool_turn,
)


@pytest.fixture
def image():
    output = io.BytesIO()
    Image.new("RGB", (30, 20), "blue").save(output, format="PNG")
    return prepare_image(output.getvalue())


async def test_codex_tool_image_reaches_next_responses_request(image):
    session, client = _session([_tool_turn("lookup", "{}"), _text_turn("blue")])
    from backend.integrations.codex.models import CodexDynamicToolResult

    async def handler(call):
        return CodexDynamicToolResult(
            content="Viewed image",
            image_urls=[image.openai()["image_url"]["url"]],
        )

    await session.invoke(_request(), [TOOL], handler)
    items = client.responses.requests[1]["input"]
    assert items[-2]["type"] == "function_call_output"
    assert items[-1]["content"][-1] == {
        "type": "input_image",
        "image_url": image.openai()["image_url"]["url"],
        "detail": "auto",
    }
    assert image.data_base64 not in items[-2]["output"]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "source",
    [
        None,
        {"type": "base64", "media_type": "image/svg+xml", "data": "unsupported"},
        {"type": "base64", "media_type": "image/png", "data": ""},
        {"type": "url", "url": "file:///private.png"},
        {"type": "url", "url": "https://example.com/image.png"},
    ],
)
async def test_shared_codex_gateway_reports_unreadable_images_before_the_upstream(
    source, nested
):
    import httpx
    from aiohttp import ClientSession

    from backend.copilot.sdk.model_gateway.adapters.codex_responses import (
        CodexResponses,
    )
    from backend.copilot.sdk.model_gateway.front import ModelGateway
    from backend.copilot.sdk.model_gateway.testing.codex_lease import codex_route

    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(500)

    content = [{"type": "image", "source": source}]
    if nested:
        content = [
            {"type": "tool_result", "tool_use_id": "old-call", "content": content}
        ]
    route = codex_route("gpt-5.6-sol")
    async with httpx.AsyncClient(
        base_url="https://codex.test", transport=httpx.MockTransport(respond)
    ) as client:
        async with ModelGateway(
            route, upstream=CodexResponses(route, client=client)
        ) as gateway:
            async with ClientSession() as incoming:
                async with incoming.post(
                    f"{gateway.base_url}/v1/messages",
                    json={"messages": [{"role": "user", "content": content}]},
                    headers={"x-api-key": gateway.auth_token},
                ) as response:
                    assert response.status == 400
                    assert "Image not shown" in await response.text()
    assert calls == []
