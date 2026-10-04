"""Unit tests for the Google Maps Platform docs blocks' MCP calls and parsing.

The blocks' own test_input/test_mock cases mock the MCP call away; these cover
what those mocks skip. The reply shapes match what the live server returned.
"""

import json

import pytest

from backend.blocks.google._maps_code_assist_api import (
    MapsCodeAssistError,
    call_maps_code_assist,
    clean_text,
    to_maps_passage,
)
from backend.blocks.google.maps_platform_docs import (
    GetGoogleMapsPlatformCodingInstructionsBlock,
    SearchGoogleMapsPlatformDocsBlock,
)
from backend.util.exceptions import BlockExecutionError
from backend.util.request import Requests

MCP_URL = "https://mapscodeassist.googleapis.com/mcp"


class _Response:
    def __init__(self, status: int, body: dict | str, content_type: str):
        self.status = status
        self.headers = {"content-type": content_type}
        self._body = body if isinstance(body, str) else json.dumps(body)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def text(self) -> str:
        return self._body


def _patch_post(monkeypatch, response: _Response) -> list[dict]:
    calls: list[dict] = []

    async def fake_post(self, url, **kwargs):
        calls.append({"url": url, "attempts": self.retry_max_attempts, **kwargs})
        return response

    monkeypatch.setattr(Requests, "post", fake_post)
    return calls


def _json(body: dict, status: int = 200) -> _Response:
    return _Response(status, body, "application/json; charset=UTF-8")


_CONTEXTS = {"contexts": [{"text": "Hi", "score": 0.7}]}


@pytest.mark.asyncio
async def test_call_sends_one_tools_call_request(monkeypatch):
    calls = _patch_post(
        monkeypatch,
        _json({"jsonrpc": "2.0", "id": 1, "result": {"structuredContent": _CONTEXTS}}),
    )
    result = await call_maps_code_assist(
        "retrieve-google-maps-platform-docs", {"llmQuery": "q"}
    )
    assert result == _CONTEXTS
    assert calls[0]["url"] == MCP_URL
    assert calls[0]["headers"] == {"Accept": "application/json, text/event-stream"}
    assert calls[0]["json"] == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "retrieve-google-maps-platform-docs",
            "arguments": {"llmQuery": "q"},
        },
    }


@pytest.mark.asyncio
async def test_call_reads_results_sent_as_text_only(monkeypatch):
    _patch_post(
        monkeypatch,
        _json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(_CONTEXTS)}]
                },
            }
        ),
    )
    assert await call_maps_code_assist("tool", {}) == _CONTEXTS


@pytest.mark.asyncio
async def test_call_reads_server_sent_event_replies(monkeypatch):
    reply = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"structuredContent": _CONTEXTS}}
    )
    _patch_post(
        monkeypatch,
        _Response(200, f"event: message\ndata: {reply}\n\n", "text/event-stream"),
    )
    assert await call_maps_code_assist("tool", {}) == _CONTEXTS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, expected",
    [
        (
            _json(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": "Request contains an invalid argument.",
                            }
                        ],
                        "isError": True,
                    },
                }
            ),
            "couldn't answer: Request contains an invalid argument.",
        ),
        (
            _json(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {
                        "code": -32602,
                        "message": "Tools Call name is not found",
                    },
                }
            ),
            "error: Tools Call name is not found",
        ),
        (_json({}, status=503), "HTTP 503. Try again later."),
        (_json({}, status=400), "HTTP 400."),
        (_Response(200, "<html></html>", "text/html"), "unreadable reply"),
        (
            _json({"jsonrpc": "2.0", "id": 1, "result": {"content": []}}),
            "unreadable result",
        ),
    ],
)
async def test_call_errors(monkeypatch, response: _Response, expected: str):
    _patch_post(monkeypatch, response)
    with pytest.raises(MapsCodeAssistError) as exc_info:
        await call_maps_code_assist("tool", {})
    assert expected in str(exc_info.value)


def test_clean_text_unescapes_documentation_passages():
    escaped = "Add a marker:\\n\\n```js\\ndocument.querySelector(\\'gmp-map\\');\\n```"
    assert clean_text(escaped) == (
        "Add a marker:\n\n```js\ndocument.querySelector('gmp-map');\n```"
    )


def test_clean_text_keeps_escaped_backslashes_literal():
    assert clean_text("a\\\\nb\\nc") == "a\\nb\nc"


def test_clean_text_leaves_passages_with_real_line_breaks_alone():
    code = 'console.log("a\\nb");\nconsole.log("c");'
    assert clean_text(code) == code
    assert clean_text("no escapes here") == "no escapes here"


def test_to_maps_passage_adds_a_scheme_to_bare_links():
    passage = to_maps_passage(
        {
            "text": "Text",
            "score": 0.77,
            "documentationUri": "developers.google.com/maps/documentation/javascript/markers",
            "apiState": "LEGACY",
        }
    )
    assert (
        passage.url
        == "https://developers.google.com/maps/documentation/javascript/markers"
    )
    assert (passage.relevance_score, passage.api_state) == (0.77, "LEGACY")

    github = to_maps_passage(
        {"text": "x", "documentationUri": "https://github.com/googlemaps/js-samples"}
    )
    assert github.url == "https://github.com/googlemaps/js-samples"
    assert to_maps_passage({}).url is None


async def _outputs(block, input_data) -> dict[str, list]:
    outputs: dict[str, list] = {}
    async for name, value in block.run(input_data):
        outputs.setdefault(name, []).append(value)
    return outputs


@pytest.mark.asyncio
async def test_search_sends_the_query_filter_and_source():
    block = SearchGoogleMapsPlatformDocsBlock()
    sent: list[dict] = []

    async def retrieve(arguments: dict) -> dict:
        sent.append(arguments)
        return {"contexts": [{"text": "A"}, {"text": "B"}]}

    block._retrieve = retrieve
    outputs = await _outputs(
        block,
        SearchGoogleMapsPlatformDocsBlock.Input(
            query="place details", product_filter=" Places API "
        ),
    )
    assert sent == [
        {
            "llmQuery": "place details",
            "source": "autogpt-platform",
            "filter": "Places API",
        }
    ]
    assert [p.text for p in outputs["results"][0]] == ["A", "B"]
    assert len(outputs["result"]) == 2


@pytest.mark.asyncio
async def test_search_reports_mcp_errors_as_block_errors():
    block = SearchGoogleMapsPlatformDocsBlock()

    async def retrieve(arguments: dict) -> dict:
        raise MapsCodeAssistError("Google Maps Code Assist returned HTTP 503.")

    block._retrieve = retrieve
    with pytest.raises(BlockExecutionError, match="HTTP 503"):
        await _outputs(block, SearchGoogleMapsPlatformDocsBlock.Input(query="q"))


@pytest.mark.asyncio
async def test_instructions_join_the_sections_and_ask_for_the_right_resource(
    monkeypatch,
):
    calls = _patch_post(
        monkeypatch,
        _json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "structuredContent": {"systemInstructions": ["One", "", "Two"]}
                },
            }
        ),
    )
    block = GetGoogleMapsPlatformCodingInstructionsBlock()
    outputs = await _outputs(
        block, GetGoogleMapsPlatformCodingInstructionsBlock.Input()
    )
    assert outputs["instructions"] == ["One\n\nTwo"]
    assert calls[0]["json"]["params"] == {
        "name": "retrieve-instructions",
        "arguments": {"name": "instructions"},
    }
