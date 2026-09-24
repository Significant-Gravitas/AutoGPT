from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ._test_data import make_session
from .models import WebFetchResponse
from .web_fetch import (
    _MAX_DOWNLOAD_BYTES,
    WebFetchTool,
    _extract_title,
    _is_client_rendered_shell,
)


def test_extract_title_decodes_entities_and_normalizes_whitespace():
    assert (
        _extract_title("<html><title> AutoGPT &amp;\n  Friends </title></html>")
        == "AutoGPT & Friends"
    )


def test_extract_title_returns_none_when_missing():
    assert _extract_title("<html><body>No title</body></html>") is None


def test_is_client_rendered_shell_detects_root_div():
    html = '<html><head><script src="/app.js"></script></head><body><div id="root"></div></body></html>'
    assert _is_client_rendered_shell(html, "Home Menu Contact") is True


def test_is_client_rendered_shell_false_for_normal_content():
    html = (
        "<html><body>"
        + ("<p>Detailed documentation section.</p>" * 30)
        + "</body></html>"
    )
    text = "Detailed documentation section. " * 30
    assert _is_client_rendered_shell(html, text) is False


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_returns_page_metadata_and_truncation():
    response = MagicMock()
    response.headers = {"content-type": "text/html; charset=utf-8"}
    response.content = (
        b"<html><title>Example</title><body>Hello</body></html>"
        + b" " * (_MAX_DOWNLOAD_BYTES + 100)
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
    assert "truncated" in result.message


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_extracts_content_below_fold_past_100kb_head():
    # Construct an HTML document with 120 KB of style/script head noise,
    # followed by the critical body content
    large_head = "<style>" + ("/* css rule */\n" * 7000) + "</style>"
    body_content = "### Enterprise Pricing Table\nTier 1: $100/mo\nTier 2: $500/mo"
    html = f"<html><head><title>Docs</title>{large_head}</head><body><h1>Docs</h1><p>{body_content}</p></body></html>"

    response = MagicMock()
    response.headers = {"content-type": "text/html; charset=utf-8"}
    response.content = html.encode("utf-8")
    response.url = "https://docs.example.com/pricing"
    response.status = 200

    client = MagicMock()
    client.get = AsyncMock(return_value=response)

    with patch("backend.copilot.tools.web_fetch.Requests", return_value=client):
        result = await WebFetchTool()._execute(
            user_id="test-user",
            session=make_session(user_id="test-user"),
            url="https://docs.example.com/pricing",
            extract_text=True,
        )

    assert isinstance(result, WebFetchResponse)
    assert result.title == "Docs"
    # Verify content below the 100 KB fold survived extraction
    assert "Enterprise Pricing Table" in result.content
    assert "Tier 1: $100/mo" in result.content
    assert "/* css rule */" not in result.content  # Head bloat was cleanly stripped


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_detects_js_rendered_shell_and_emits_hint():
    spa_html = (
        "<!DOCTYPE html><html><head><title>SPA App</title>"
        '<script src="/bundle.js"></script></head>'
        '<body><div id="root"></div>'
        '<nav><a href="/home">Home</a><a href="/login">Login</a></nav>'
        "</body></html>"
    )

    response = MagicMock()
    response.headers = {"content-type": "text/html; charset=utf-8"}
    response.content = spa_html.encode("utf-8")
    response.url = "https://app.example.com"
    response.status = 200

    client = MagicMock()
    client.get = AsyncMock(return_value=response)

    with patch("backend.copilot.tools.web_fetch.Requests", return_value=client):
        result = await WebFetchTool()._execute(
            user_id="test-user",
            session=make_session(user_id="test-user"),
            url="https://app.example.com",
            extract_text=True,
        )

    assert isinstance(result, WebFetchResponse)
    assert "browser_navigate" in result.message
    assert "browser_navigate" in result.content
    assert "Content not rendered" in result.content


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
