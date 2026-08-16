"""Tests for JSON content-type fix in SendWebRequestBlock (issue #14007)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.http import HttpMethod, SendWebRequestBlock
from backend.data.execution import ExecutionContext
from backend.util.request import Response


def make_test_context(graph_exec_id="test-exec", user_id="test-user"):
    return ExecutionContext(user_id=user_id, graph_exec_id=graph_exec_id)


def make_response(status, content_type, json_data=None, text_data=""):
    resp = MagicMock(spec=Response)
    resp.status = status
    resp.headers = {"content-type": content_type}
    if json_data is not None:
        resp.json.return_value = json_data
    resp.text.return_value = text_data
    return resp


class TestJsonContentTypeParsing:
    @pytest.mark.asyncio
    @patch("backend.blocks.http.Requests")
    async def test_plus_json_suffix_parsed_as_json(self, mock_req_cls):
        """application/*+json types like vnd.api+json, hal+json are parsed as JSON."""
        block = SendWebRequestBlock()
        types = [
            "application/vnd.api+json",
            "application/hal+json",
            "application/merge-patch+json",
            "application/problem+json",
            "application/json-patch+json",
        ]
        for ct in types:
            resp = make_response(200, ct, json_data={"key": "value"})
            mock_req = AsyncMock()
            mock_req.request.return_value = resp
            mock_req_cls.return_value = mock_req

        result = []
        async for name, data in block.run(
            SendWebRequestBlock.Input(
                url="https://api.example.com",
                method=HttpMethod.GET,
            ),
            execution_context=make_test_context(),
        ):
            result.append((name, data))

        assert len(result) == 1
        assert result[0] == ("response", {"key": "value"})
        resp.json.assert_called_once()
        resp.text.assert_not_called()

    @pytest.mark.asyncio
    @patch("backend.blocks.http.Requests")
    async def test_non_json_plus_suffix_returns_text(self, mock_req_cls):
        """Plain text types like text/plain should NOT be parsed as JSON."""
        block = SendWebRequestBlock()
        resp = make_response(200, "text/plain", text_data="hello world")
        mock_req = AsyncMock()
        mock_req.request.return_value = resp
        mock_req_cls.return_value = mock_req

        result = []
        async for name, data in block.run(
            SendWebRequestBlock.Input(
                url="https://api.example.com",
                method=HttpMethod.GET,
            ),
            execution_context=make_test_context(),
        ):
            result.append((name, data))

        assert len(result) == 1
        assert result[0] == ("response", "hello world")
        resp.text.assert_called_once()
        resp.json.assert_not_called()

    @pytest.mark.asyncio
    @patch("backend.blocks.http.Requests")
    async def test_plus_json_204_returns_none(self, mock_req_cls):
        """204 with +json content type should return None, not try to parse body."""
        block = SendWebRequestBlock()
        resp = make_response(204, "application/vnd.api+json")
        mock_req = AsyncMock()
        mock_req.request.return_value = resp
        mock_req_cls.return_value = mock_req

        result = []
        async for name, data in block.run(
            SendWebRequestBlock.Input(
                url="https://api.example.com",
                method=HttpMethod.GET,
            ),
            execution_context=make_test_context(),
        ):
            result.append((name, data))

        assert len(result) == 1
        assert result[0] == ("response", None)
        resp.json.assert_not_called()
        resp.text.assert_not_called()
