"""Unit tests for the Websets HTTP helper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.exa._exa_rest import WEBSETS_API_BASE, ExaApiError
from backend.blocks.exa._test import TEST_CREDENTIALS
from backend.blocks.exa.websets import ExaGetWebsetBlock
from backend.blocks.exa.websets_api import query_params, segment


def _response(status: int, body: bytes) -> MagicMock:
    response = MagicMock()
    response.ok = 200 <= status < 300
    response.status = status
    response.content = body
    response.text.return_value = body.decode()
    response.json.return_value = {"id": "webset_1"} if body else {}
    return response


@pytest.mark.asyncio
async def test_api_targets_websets_base_with_api_key():
    block = ExaGetWebsetBlock()
    with patch("backend.blocks.exa._exa_rest.Requests") as requests_cls:
        requests_cls.return_value.request = AsyncMock(
            return_value=_response(200, b'{"id": "webset_1"}')
        )
        data = await block._api(
            TEST_CREDENTIALS,
            "GET",
            "/websets",
            params={"limit": 5, "search": None, "flag": True},
        )

    assert data == {"id": "webset_1"}
    method, url = requests_cls.return_value.request.call_args.args
    kwargs = requests_cls.return_value.request.call_args.kwargs
    assert (method, url) == ("GET", f"{WEBSETS_API_BASE}/websets")
    assert url == "https://api.exa.ai/websets/v0/websets"
    assert kwargs["headers"]["x-api-key"] == "mock-exa-api-key"
    assert kwargs["params"] == [("limit", "5"), ("flag", "true")]


@pytest.mark.asyncio
async def test_api_error_carries_exa_message_and_status():
    block = ExaGetWebsetBlock()
    body = b'{"requestId": "r", "error": "Webset not found", "tag": "NOT_FOUND"}'
    with patch("backend.blocks.exa._exa_rest.Requests") as requests_cls:
        requests_cls.return_value.request = AsyncMock(return_value=_response(404, body))
        with pytest.raises(ExaApiError, match="Webset not found") as exc_info:
            await block._api(TEST_CREDENTIALS, "GET", "/websets/missing")

    assert exc_info.value.status == 404


@pytest.mark.asyncio
async def test_api_empty_body_returns_empty_dict():
    block = ExaGetWebsetBlock()
    with patch("backend.blocks.exa._exa_rest.Requests") as requests_cls:
        requests_cls.return_value.request = AsyncMock(return_value=_response(204, b""))
        assert await block._api(TEST_CREDENTIALS, "DELETE", "/imports/i") == {}


def test_query_params_drop_unset_values_and_render_booleans():
    assert query_params({"a": 1, "b": None, "c": False, "d": ""}) == [
        ("a", "1"),
        ("c", "false"),
    ]
    assert query_params(None) == []


def test_segment_keeps_external_ids_inside_one_path_segment():
    assert segment("team/a b?x") == "team%2Fa%20b%3Fx"
