"""Tests for JSON content-type parsing in SendWebRequestBlock (issue #14007)."""

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
        """application/*+json types are parsed as JSON, including parameters."""
        block = SendWebRequestBlock()
        content_types = [
            "application/vnd.api+json",
            "application/hal+json",
            "application/merge-patch+json",
            "application/problem+json; charset=utf-8",
            "Application/JSON-Patch+JSON; Charset=UTF-8",
        ]

        for content_type in content_types:
            resp = make_response(200, content_type, json_data={"key": "value"})
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

            assert result == [("response", {"key": "value"})]
            resp.json.assert_called_once()
            resp.text.assert_not_called()

    @pytest.mark.asyncio
    @patch("backend.blocks.http.Requests")
    async def test_non_json_plus_suffix_returns_text(self, mock_req_cls):
        """Plain text types like text/plain should not be parsed as JSON."""
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

        assert result == [("response", "hello world")]
        resp.text.assert_called_once()
        resp.json.assert_not_called()

    @pytest.mark.asyncio
    @patch("backend.blocks.http.Requests")
    async def test_plus_json_204_returns_none(self, mock_req_cls):
        """204 with +json content type returns None without parsing a body."""
        block = SendWebRequestBlock()
        resp = make_response(204, "application/vnd.api+json; charset=utf-8")
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

        assert result == [("response", None)]
        resp.json.assert_not_called()
        resp.text.assert_not_called()
