"""Unit tests for the Google Search Console List Sites and Get Performance blocks.

The blocks' own test_input/test_mock cases mock the API calls away. These run
the blocks against the real Search Console client, built from the discovery
document the client library ships with, and replace only the HTTP layer with
canned responses. So they check the requests the blocks send, how they read
the responses and how they report errors.
"""

import json
from datetime import date
from typing import Any

import pytest
from googleapiclient.discovery import build
from googleapiclient.http import HttpMockSequence

from backend.blocks.google import _search_console_inputs, search_console
from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google._search_console_models import (
    SearchConsoleRow,
    SearchConsoleSite,
)
from backend.blocks.google._search_console_testdata import (
    TEST_ANALYTICS_RESPONSE,
    TEST_SITE_ENTRIES,
)
from backend.blocks.google.search_console import (
    GoogleSearchConsoleGetPerformanceBlock,
    GoogleSearchConsoleListSitesBlock,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError

API = "https://searchconsole.googleapis.com/webmasters/v3"


@pytest.fixture
def search_console_api(monkeypatch: pytest.MonkeyPatch):
    """Serve canned responses to the blocks' Search Console client."""

    def install(*responses: tuple[int, dict[str, Any]]) -> HttpMockSequence:
        http = HttpMockSequence(
            [({"status": str(status)}, json.dumps(body)) for status, body in responses]
        )
        service = build("searchconsole", "v1", http=http, cache_discovery=False)
        monkeypatch.setattr(
            search_console, "build_search_console_service", lambda _: service
        )
        return http

    return install


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        _search_console_inputs, "pacific_today", lambda: date(2026, 10, 5)
    )


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


async def test_list_sites(search_console_api):
    http = search_console_api((200, {"siteEntry": TEST_SITE_ENTRIES}))
    outputs = await _run(GoogleSearchConsoleListSitesBlock())
    assert _requests(http) == [("GET", f"{API}/sites?alt=json", None)]
    sites = [
        SearchConsoleSite(
            site_url="sc-domain:example.com", permission_level="siteOwner"
        ),
        SearchConsoleSite(
            site_url="https://www.example.com/", permission_level="siteFullUser"
        ),
        SearchConsoleSite(
            site_url="https://shop.example.org/", permission_level="siteUnverifiedUser"
        ),
    ]
    assert outputs == [("sites", sites), *(("site", site) for site in sites)]


async def test_list_sites_for_an_account_without_properties(search_console_api):
    search_console_api((200, {}))
    assert await _run(GoogleSearchConsoleListSitesBlock()) == [("sites", [])]


async def test_list_sites_explains_a_disabled_api(search_console_api):
    search_console_api(
        _error(
            403,
            "Google Search Console API has not been used in project 1 before or it "
            "is disabled.",
            "PERMISSION_DENIED",
        )
    )
    with pytest.raises(BlockExecutionError, match="isn't enabled for this AutoGPT"):
        await _run(GoogleSearchConsoleListSitesBlock())


async def test_performance_request_for_a_property(search_console_api):
    http = search_console_api((200, TEST_ANALYTICS_RESPONSE))
    outputs = await _run(
        GoogleSearchConsoleGetPerformanceBlock(),
        site_url="https://www.example.com/",
        dimensions=["query", "page"],
    )
    assert _requests(http) == [
        (
            "POST",
            f"{API}/sites/https%3A%2F%2Fwww.example.com%2F/searchAnalytics/query?alt=json",
            {
                "startDate": "2026-09-05",
                "endDate": "2026-10-02",
                "dimensions": ["query", "page"],
                "type": "web",
                "dataState": "final",
                "rowLimit": 100,
                "startRow": 0,
            },
        )
    ]
    rows = [
        SearchConsoleRow(
            query="running shoes",
            page="https://www.example.com/shoes/",
            clicks=120,
            impressions=3400,
            ctr=0.0353,
            position=4.2,
        ),
        SearchConsoleRow(
            query="trail shoes",
            page="https://www.example.com/trail/",
            clicks=45,
            impressions=2100,
            ctr=0.0214,
            position=7.8,
        ),
    ]
    assert outputs == [
        ("rows", rows),
        ("row", rows[0]),
        ("row", rows[1]),
        ("site_url", "https://www.example.com/"),
    ]


async def test_performance_finds_the_property_for_a_bare_domain(search_console_api):
    http = search_console_api(
        (
            200,
            {
                "siteEntry": [
                    {"siteUrl": "https://example.com/", "permissionLevel": "siteOwner"},
                    {
                        "siteUrl": "sc-domain:example.com",
                        "permissionLevel": "siteRestrictedUser",
                    },
                ]
            },
        ),
        (200, {"rows": []}),
    )
    outputs = await _run(
        GoogleSearchConsoleGetPerformanceBlock(), site_url="www.example.com"
    )
    assert [(method, uri) for method, uri, _ in _requests(http)] == [
        ("GET", f"{API}/sites?alt=json"),
        ("POST", f"{API}/sites/sc-domain%3Aexample.com/searchAnalytics/query?alt=json"),
    ]
    assert outputs == [("rows", []), ("site_url", "sc-domain:example.com")]


async def test_performance_with_filters_fresh_data_and_no_dimensions(
    search_console_api,
):
    http = search_console_api(
        (200, {"rows": [{"clicks": 1520, "impressions": 48210, "ctr": 0.0315}]})
    )
    outputs = await _run(
        GoogleSearchConsoleGetPerformanceBlock(),
        site_url="sc-domain:example.com",
        start_date="yesterday",
        end_date="today",
        dimensions=[],
        search_type="discover",
        filters=[{"dimension": "country", "operator": "equals", "expression": "usa"}],
        row_limit=10,
        start_row=10,
        include_fresh_data=True,
    )
    assert _requests(http)[0][2] == {
        "startDate": "2026-10-04",
        "endDate": "2026-10-05",
        "dimensions": [],
        "type": "discover",
        "dataState": "all",
        "rowLimit": 10,
        "startRow": 10,
        "dimensionFilterGroups": [
            {
                "groupType": "and",
                "filters": [
                    {"dimension": "country", "operator": "equals", "expression": "usa"}
                ],
            }
        ],
    }
    totals = SearchConsoleRow(clicks=1520, impressions=48210, ctr=0.0315)
    assert outputs == [
        ("rows", [totals]),
        ("row", totals),
        ("site_url", "sc-domain:example.com"),
    ]


async def test_performance_asks_for_each_dimension_once(search_console_api):
    http = search_console_api(
        (200, {"rows": [{"keys": ["usa", "MOBILE"], "clicks": 2, "impressions": 9}]})
    )
    outputs = dict(
        await _run(
            GoogleSearchConsoleGetPerformanceBlock(),
            site_url="sc-domain:example.com",
            dimensions=["country", "device", "country"],
        )
    )
    assert _requests(http)[0][2]["dimensions"] == ["country", "device"]
    assert outputs["row"] == SearchConsoleRow(
        country="usa", device="MOBILE", clicks=2, impressions=9, ctr=0.0
    )


async def test_performance_checks_dates_before_calling_google(search_console_api):
    http = search_console_api()
    with pytest.raises(BlockInputError, match="start_date 'last month' isn't a date"):
        await _run(
            GoogleSearchConsoleGetPerformanceBlock(),
            site_url="sc-domain:example.com",
            start_date="last month",
        )
    assert http.request_sequence == []


def test_performance_input_needs_a_filter_expression():
    error = GoogleSearchConsoleGetPerformanceBlock.Input.validate_data(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "site_url": "sc-domain:example.com",
            "filters": [
                {"dimension": "query", "operator": "contains", "expression": ""}
            ],
        }
    )
    assert error


async def test_performance_explains_a_property_the_account_cant_read(
    search_console_api,
):
    search_console_api(
        _error(
            403,
            "User does not have sufficient permission for site "
            "'sc-domain:example.org'. See also: "
            "https://support.google.com/webmasters/answer/2451999.",
            "PERMISSION_DENIED",
        )
    )
    with pytest.raises(
        BlockExecutionError,
        match="can't read the Search Console property sc-domain:example.org",
    ):
        await _run(
            GoogleSearchConsoleGetPerformanceBlock(), site_url="sc-domain:example.org"
        )


async def test_performance_passes_googles_bad_request_message_on(search_console_api):
    search_console_api(
        _error(400, "Request contains an invalid argument.", "INVALID_ARGUMENT")
    )
    with pytest.raises(
        BlockExecutionError,
        match="rejected the request: Request contains an invalid argument.",
    ):
        await _run(
            GoogleSearchConsoleGetPerformanceBlock(),
            site_url="sc-domain:example.com",
            search_type="discover",
        )
