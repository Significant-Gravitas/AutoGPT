from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ._test_data import make_session
from .models import WebFetchResponse
from .web_fetch import WebFetchTool, _extract_title


def test_extract_title_decodes_entities_and_normalizes_whitespace():
    assert (
        _extract_title("<html><title> AutoGPT &amp;\n  Friends </title></html>")
        == "AutoGPT & Friends"
    )


def test_extract_title_returns_none_when_missing():
    assert _extract_title("<html><body>No title</body></html>") is None


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_returns_page_metadata_and_truncation():
    response = MagicMock()
    response.headers = {"content-type": "text/html; charset=utf-8"}
    response.content = (
        b"<html><title>Example</title><body>Hello</body></html>" + b" " * 102_400
    )
    response.url = "https://example.com/"
    response.status = 200
    client = MagicMock()
    client.get = AsyncMock(return_value=response)

    with patch("backend.copilot.tools.web_fetch.Requests", return_value=client):
        result = await WebFetchTool()._execute(
            user_id="test-user",
            session=make_session(user_id="test-user"),
            url="https://example.com",
        )

    assert isinstance(result, WebFetchResponse)
    assert result.title == "Example"
    assert result.content_length == len(response.content)
    assert result.truncated is True


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_reports_small_original_body_size_after_text_extraction():
    response = MagicMock()
    response.headers = {"content-type": "text/html; charset=utf-8"}
    response.content = b"<html><title>Example</title><body>Hello</body></html>"
    response.url = "https://example.com/"
    response.status = 200
    client = MagicMock()
    client.get = AsyncMock(return_value=response)

    with patch("backend.copilot.tools.web_fetch.Requests", return_value=client):
        result = await WebFetchTool()._execute(
            user_id="test-user",
            session=make_session(user_id="test-user"),
            url="https://example.com",
            extract_text=True,
        )

    assert isinstance(result, WebFetchResponse)
    assert "Hello" in result.content
    assert "<body>" not in result.content
    assert result.content_length == len(response.content)
    assert result.truncated is False
