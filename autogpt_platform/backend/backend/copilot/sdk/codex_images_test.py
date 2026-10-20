import io
import json

import pytest
from aiohttp import ClientSession
from PIL import Image

from backend.copilot.sdk.codex_compat_gateway import (
    CodexAnthropicGateway,
    _estimate_input_tokens,
    _extract_tool_results,
    _serialize_messages,
)
from backend.copilot.sdk.codex_compat_gateway_test import (
    _FakeAgentSession,
    _FakeTransport,
    _headers,
    _lease,
)
from backend.copilot.sdk.codex_images import image_urls_from_content
from backend.copilot.tool_image import prepare_image
from backend.integrations.codex.http_session_test import (
    TOOL,
    _echo_tool,
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
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": [
                        {"type": "text", "text": "Viewed image"},
                        image.anthropic(),
                    ],
                }
            ],
        }
    ]

    async def handler(call):
        return _extract_tool_results(messages)[call.call_id]

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
async def test_codex_initial_and_replayed_images_are_not_json_text(image, nested):
    content = [image.anthropic()]
    if nested:
        content = [
            {"type": "tool_result", "tool_use_id": "old-call", "content": content}
        ]
    image_urls = []
    prompt = _serialize_messages([{"role": "user", "content": content}], image_urls)
    assert image.data_base64 not in prompt
    assert "Image 1" in prompt
    session, client = _session([_text_turn("blue")])
    await session.invoke(_request(prompt=prompt, image_urls=image_urls), [], _echo_tool)
    assert client.responses.requests[0]["input"][0]["content"][-1] == {
        "type": "input_image",
        "image_url": image.openai()["image_url"]["url"],
        "detail": "auto",
    }
    assert "image input omitted" not in json.dumps(client.responses.requests)


async def test_gateway_passes_initial_image_to_transport(image):
    session = _FakeAgentSession()
    async with CodexAnthropicGateway(
        transport=_FakeTransport(session), credential_lease=_lease(), model="gpt-5.4"
    ) as gateway, ClientSession() as client:
        response = await client.post(
            f"{gateway.base_url}/v1/messages",
            headers=_headers(gateway),
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image"},
                            image.anthropic(),
                        ],
                    }
                ]
            },
        )
        await response.read()
        assert response.status == 200
    assert session.requests[0].image_urls == [image.openai()["image_url"]["url"]]
    assert image.data_base64 not in session.requests[0].prompt


@pytest.mark.parametrize(
    "source",
    [
        None,
        {"type": "base64", "media_type": "image/svg+xml", "data": "unsupported"},
        {"type": "base64", "media_type": "image/png", "data": ""},
        {"type": "url", "url": "file:///private.png"},
    ],
)
def test_unsupported_gateway_image_is_reported_instead_of_omitted(source):
    with pytest.raises(ValueError, match="Image not shown"):
        image_urls_from_content([{"type": "image", "source": source}])


async def test_gateway_reports_unreadable_initial_image():
    session = _FakeAgentSession()
    async with CodexAnthropicGateway(
        transport=_FakeTransport(session), credential_lease=_lease(), model="gpt-5.4"
    ) as gateway, ClientSession() as client:
        response = await client.post(
            f"{gateway.base_url}/v1/messages",
            headers=_headers(gateway),
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/svg+xml",
                                    "data": "unsupported",
                                },
                            },
                        ],
                    }
                ]
            },
        )
        body = await response.text()
        assert response.status == 400
        assert "Image not shown" in body
    assert not session.requests


def test_image_token_estimate_does_not_count_base64_as_text(image):
    small = image.anthropic()
    large = image.anthropic()
    large["source"]["data"] = "A" * (4 * 1024 * 1024)

    def payload(block):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": [block],
                        },
                    ],
                }
            ]
        }

    assert _estimate_input_tokens(payload(small)) == _estimate_input_tokens(
        payload(large)
    )
    assert 1000 < _estimate_input_tokens(payload(large)) < 10000
