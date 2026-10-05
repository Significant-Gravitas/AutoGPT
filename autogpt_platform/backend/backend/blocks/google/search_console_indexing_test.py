"""Unit tests for the Google Search Console Inspect URL and List Sitemaps blocks.

The blocks' own test_input/test_mock cases mock the API calls away. These run
the blocks against the real Search Console client, built from the discovery
document the client library ships with, and replace only the HTTP layer with
canned responses. So they check the requests the blocks send, how they read
the responses and how they report errors.
"""

import json
from typing import Any

import pytest
from googleapiclient.discovery import build
from googleapiclient.http import HttpMockSequence

from backend.blocks.google import search_console_indexing
from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google._search_console_models import (
    SearchConsoleSitemap,
    SearchConsoleSitemapContent,
)
from backend.blocks.google._search_console_testdata import (
    TEST_INSPECTED_URL,
    TEST_INSPECTION_RESULT,
    TEST_SITEMAPS_RESPONSE,
)
from backend.blocks.google.search_console_indexing import (
    GoogleSearchConsoleInspectURLBlock,
    GoogleSearchConsoleListSitemapsBlock,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError

API = "https://searchconsole.googleapis.com"


@pytest.fixture
def search_console_api(monkeypatch: pytest.MonkeyPatch):
    """Serve canned responses to the blocks' Search Console client."""

    def install(*responses: tuple[int, dict[str, Any]]) -> HttpMockSequence:
        http = HttpMockSequence(
            [({"status": str(status)}, json.dumps(body)) for status, body in responses]
        )
        service = build("searchconsole", "v1", http=http, cache_discovery=False)
        monkeypatch.setattr(
            search_console_indexing, "build_search_console_service", lambda _: service
        )
        return http

    return install


def _requests(http: HttpMockSequence) -> list[tuple[str, str, Any]]:
    """Each request the client sent, as (method, URI, JSON body)."""
    return [
        (method, uri, json.loads(body) if body else None)
        for uri, method, body, _ in http.request_sequence
    ]


def _error(status: int, message: str, api_status: str) -> tuple[int, dict[str, Any]]:
    return status, {"error": {"code": status, "message": message, "status": api_status}}


async def _run(block, **fields) -> list[tuple[str, Any]]:
    input_data = block.input_schema.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **fields}
    )
    return [
        output async for output in block.run(input_data, credentials=TEST_CREDENTIALS)
    ]


async def test_inspect_url_request_and_outputs(search_console_api):
    http = search_console_api((200, {"inspectionResult": TEST_INSPECTION_RESULT}))
    outputs = await _run(
        GoogleSearchConsoleInspectURLBlock(),
        inspection_url=TEST_INSPECTED_URL,
        site_url="sc-domain:example.com",
    )
    assert _requests(http) == [
        (
            "POST",
            f"{API}/v1/urlInspection/index:inspect?alt=json",
            {
                "inspectionUrl": TEST_INSPECTED_URL,
                "siteUrl": "sc-domain:example.com",
                "languageCode": "en-US",
            },
        )
    ]
    assert [name for name, _ in outputs] == [
        "verdict",
        "coverage_state",
        "indexing_state",
        "robots_txt_state",
        "page_fetch_state",
        "last_crawl_time",
        "crawled_as",
        "google_canonical",
        "user_canonical",
        "sitemaps",
        "referring_urls",
        "rich_results_verdict",
        "rich_result_types",
        "inspection_result_link",
        "inspection_result",
        "site_url",
    ]
    values = dict(outputs)
    assert values["coverage_state"] == "Submitted and indexed"
    assert values["inspection_result"] == TEST_INSPECTION_RESULT
    assert values["site_url"] == "sc-domain:example.com"


async def test_inspect_url_picks_a_property_that_holds_the_url(search_console_api):
    http = search_console_api(
        (
            200,
            {
                "siteEntry": [
                    {"siteUrl": "https://example.com/", "permissionLevel": "siteOwner"},
                    {
                        "siteUrl": "https://www.example.com/",
                        "permissionLevel": "siteFullUser",
                    },
                ]
            },
        ),
        (200, {"inspectionResult": {}}),
    )
    outputs = await _run(
        GoogleSearchConsoleInspectURLBlock(),
        inspection_url=TEST_INSPECTED_URL,
        site_url="example.com",
        language_code=" ",
    )
    requests = _requests(http)
    assert requests[0][:2] == ("GET", f"{API}/webmasters/v3/sites?alt=json")
    assert requests[1][2] == {
        "inspectionUrl": TEST_INSPECTED_URL,
        "siteUrl": "https://www.example.com/",
        "languageCode": "en-US",
    }
    assert outputs == [
        ("sitemaps", []),
        ("referring_urls", []),
        ("rich_result_types", []),
        ("inspection_result", {}),
        ("site_url", "https://www.example.com/"),
    ]


async def test_inspect_url_sends_the_language_given(search_console_api):
    http = search_console_api((200, {"inspectionResult": {}}))
    await _run(
        GoogleSearchConsoleInspectURLBlock(),
        inspection_url=TEST_INSPECTED_URL,
        site_url="sc-domain:example.com",
        language_code="de-CH",
    )
    assert _requests(http)[0][2]["languageCode"] == "de-CH"


async def test_inspect_url_wants_a_full_url_before_calling_google(search_console_api):
    http = search_console_api()
    with pytest.raises(BlockInputError, match="full URL of the page"):
        await _run(
            GoogleSearchConsoleInspectURLBlock(),
            inspection_url="www.example.com/pricing",
            site_url="sc-domain:example.com",
        )
    assert http.request_sequence == []


async def test_inspect_url_explains_a_url_outside_the_property(search_console_api):
    search_console_api(
        _error(
            403,
            "You do not own this site, or the inspected URL is not part of this "
            "property.",
            "PERMISSION_DENIED",
        )
    )
    with pytest.raises(
        BlockExecutionError, match=r"or https://shop\.example\.org/a isn't part of it"
    ):
        await _run(
            GoogleSearchConsoleInspectURLBlock(),
            inspection_url="https://shop.example.org/a",
            site_url="sc-domain:example.com",
        )


async def test_inspect_url_explains_the_inspection_quota(search_console_api):
    search_console_api(
        _error(
            429,
            "Quota exceeded for quota metric 'URL inspection requests'.",
            "RESOURCE_EXHAUSTED",
        )
    )
    with pytest.raises(BlockExecutionError, match="2,000 inspections a day"):
        await _run(
            GoogleSearchConsoleInspectURLBlock(),
            inspection_url=TEST_INSPECTED_URL,
            site_url="sc-domain:example.com",
        )


async def test_list_sitemaps_request_and_outputs(search_console_api):
    http = search_console_api((200, TEST_SITEMAPS_RESPONSE))
    outputs = await _run(
        GoogleSearchConsoleListSitemapsBlock(), site_url="https://www.example.com"
    )
    assert _requests(http) == [
        (
            "GET",
            f"{API}/webmasters/v3/sites/https%3A%2F%2Fwww.example.com%2F/sitemaps?alt=json",
            None,
        )
    ]
    sitemap = SearchConsoleSitemap(
        path="https://www.example.com/sitemap.xml",
        type="sitemap",
        last_submitted="2026-08-01T10:15:00.000Z",
        last_downloaded="2026-10-03T22:41:07.512Z",
        is_pending=False,
        is_sitemaps_index=False,
        warnings=2,
        errors=0,
        contents=[
            SearchConsoleSitemapContent(type="web", submitted=1520),
            SearchConsoleSitemapContent(type="image", submitted=310),
        ],
    )
    assert outputs == [
        ("sitemaps", [sitemap]),
        ("sitemap", sitemap),
        ("site_url", "https://www.example.com/"),
    ]


async def test_list_sitemaps_inside_an_index(search_console_api):
    http = search_console_api((200, {}))
    outputs = await _run(
        GoogleSearchConsoleListSitemapsBlock(),
        site_url="sc-domain:example.com",
        sitemap_index=" https://www.example.com/sitemap_index.xml ",
    )
    assert _requests(http)[0][1] == (
        f"{API}/webmasters/v3/sites/sc-domain%3Aexample.com/sitemaps"
        "?sitemapIndex=https%3A%2F%2Fwww.example.com%2Fsitemap_index.xml&alt=json"
    )
    assert outputs == [("sitemaps", []), ("site_url", "sc-domain:example.com")]


async def test_list_sitemaps_explains_a_missing_scope(search_console_api):
    search_console_api(
        _error(
            403, "Request had insufficient authentication scopes.", "PERMISSION_DENIED"
        )
    )
    with pytest.raises(BlockExecutionError, match="Reconnect Google"):
        await _run(
            GoogleSearchConsoleListSitemapsBlock(), site_url="sc-domain:example.com"
        )
