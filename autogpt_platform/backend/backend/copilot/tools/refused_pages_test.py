"""A page that refuses an automated visitor gets a notice naming the site and a
fallback; a page that loads does not. Fixtures are shaped like what each site
served an anonymous fetch and a headless browser on 3 Nov 2026."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ._test_data import make_session
from .agent_browser import BrowserNavigateTool
from .models import BrowserNavigateResponse, ErrorResponse, WebFetchResponse
from .web_fetch import WebFetchTool

# Facebook and Instagram: a 200 whose body is all script, the summary only in <meta>.
_SCRIPT_ONLY_PAGE = (
    "<html><head><title>Example Bakery</title>"
    '<meta name="description" content="Example Bakery. 1,234 followers.">'
    "<script>requireLazy(['Bootloader'], function () {});</script></head>"
    "<body><script>window.__bbox = {};</script></body></html>"
)
_READABLE_PAGE = (
    "<html><head><title>Example Bakery (@examplebakery) / X</title></head>"
    "<body><a href='/i/flow/login'>Log in</a><p>Example Bakery</p>"
    "<p>1,234 posts</p></body></html>"
)
_CHALLENGE_PAGE = (
    "<html><head><title>Just a moment...</title></head>"
    "<body><script>window._cf_chl_opt = {};</script></body></html>"
)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "url, status, body, headers, refused_by",
    [
        (
            "https://www.facebook.com/examplebakery",
            200,
            _SCRIPT_ONLY_PAGE,
            {},
            "Facebook",
        ),
        (
            "https://m.facebook.com/examplebakery",
            200,
            _SCRIPT_ONLY_PAGE,
            {},
            "Facebook",
        ),
        (
            "https://www.instagram.com/examplebakery/",
            200,
            _SCRIPT_ONLY_PAGE,
            {},
            "Instagram",
        ),
        (
            "https://www.linkedin.com/authwall?trk=bf",
            200,
            _READABLE_PAGE,
            {},
            "LinkedIn",
        ),
        (
            "https://www.example.com/pricing",
            403,
            _CHALLENGE_PAGE,
            {"cf-mitigated": "challenge"},
            "example.com answered with a bot check",
        ),
        ("https://x.com/examplebakery", 200, _READABLE_PAGE, {}, None),
        # An empty page off the listed sites is not read as a wall.
        ("https://www.example.com/gallery", 200, _SCRIPT_ONLY_PAGE, {}, None),
        # A server error or rate limit on a listed site stays retryable.
        ("https://www.facebook.com/examplebakery", 502, _SCRIPT_ONLY_PAGE, {}, None),
        ("https://www.instagram.com/examplebakery/", 429, _SCRIPT_ONLY_PAGE, {}, None),
    ],
)
async def test_web_fetch_names_the_refusing_site_and_a_fallback(
    url, status, body, headers, refused_by
):
    response = MagicMock()
    response.headers = {"content-type": "text/html; charset=utf-8", **headers}
    response.content = body.encode()
    response.url = url
    response.status = status
    client = MagicMock()
    client.get = AsyncMock(return_value=response)

    with patch("backend.copilot.tools.web_fetch.Requests", return_value=client):
        result = await WebFetchTool()._execute(
            user_id="test-user", session=make_session(user_id="test-user"), url=url
        )

    if refused_by is None:
        assert isinstance(result, WebFetchResponse)
        assert "web_search" not in result.message
        return
    assert isinstance(result, ErrorResponse)
    assert result.error == "page_refused"
    assert refused_by in result.message
    assert "web_search" in result.message


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "url, open_result, page_check, expected_error, refused_by",
    [
        (
            "https://www.facebook.com/examplebakery",
            (0, "", ""),
            "sign_in_form",
            None,
            "Facebook",
        ),
        (
            "https://www.example.com/pricing",
            (0, "", ""),
            "bot_check",
            None,
            "example.com answered with a bot check",
        ),
        (
            "https://x.com/examplebakery",
            (1, "", "✗ Navigation failed: net::ERR_HTTP_RESPONSE_CODE_FAILURE\n"),
            "",
            "page_refused",
            "X shows",
        ),
        # A sign-in form or a failed load off the listed sites stays as it was.
        ("https://shop.example.com/login", (0, "", ""), "sign_in_form", None, None),
        (
            "https://www.example.com/",
            (1, "", "✗ Navigation failed: net::ERR_HTTP_RESPONSE_CODE_FAILURE\n"),
            "",
            "navigation_failed",
            None,
        ),
    ],
)
async def test_browser_navigate_names_the_refusing_site_and_a_fallback(
    url, open_result, page_check, expected_error, refused_by
):
    async def fake_run(session_name, *args, **kwargs):
        if args[0] == "open":
            return open_result
        if args[0] == "eval":
            return 0, f'"{page_check}"\n', ""
        if args[:2] == ("get", "url"):
            return 0, f"{url}\n", ""
        return 0, "", ""

    with (
        patch(
            "backend.copilot.tools.agent_browser.validate_url_host",
            new_callable=AsyncMock,
        ),
        patch("backend.copilot.tools.agent_browser._run", side_effect=fake_run),
        patch(
            "backend.copilot.tools.agent_browser._snapshot",
            new_callable=AsyncMock,
            return_value="- textbox [ref=e1]",
        ),
    ):
        result = await BrowserNavigateTool()._execute(
            user_id=None, session=make_session(user_id="test-user"), url=url
        )

    if expected_error:
        assert isinstance(result, ErrorResponse)
        assert result.error == expected_error
    else:
        assert isinstance(result, BrowserNavigateResponse)
    if refused_by is None:
        assert "web_search" not in result.message
    else:
        assert refused_by in result.message
        assert "web_search" in result.message
