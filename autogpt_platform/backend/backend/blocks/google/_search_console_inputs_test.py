"""Unit tests for turning the Search Console blocks' site and date inputs into
what the API takes."""

import json
from datetime import date, datetime, timezone

import httplib2
import pytest
from googleapiclient.errors import HttpError

from backend.blocks.google import _search_console_inputs
from backend.blocks.google._search_console_inputs import (
    as_property,
    match_site,
    pacific_today,
    property_contains,
    require_page_url,
    resolve_date,
    resolve_dates,
    resolve_site_url,
    site_candidates,
)
from backend.blocks.google._search_console_models import SearchConsoleSite
from backend.util.exceptions import BlockExecutionError, BlockInputError

TODAY = date(2026, 10, 5)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("today", date(2026, 10, 5)),
        ("yesterday", date(2026, 10, 4)),
        ("28daysAgo", date(2026, 9, 7)),
        (" 3DaysAgo ", date(2026, 10, 2)),
        ("0daysAgo", TODAY),
        ("Today", TODAY),
        ("2024-02-29", date(2024, 2, 29)),
    ],
)
def test_resolve_date_reads_the_google_analytics_forms(value: str, expected: date):
    assert resolve_date(value, TODAY) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "last week",
        "2026-02-30",
        "20260228",
        "2026-W01-1",
        "2026/02/28",
        "-3daysAgo",
        "3 days ago",
        "999999daysAgo",
        "99999999999daysAgo",
    ],
)
def test_resolve_date_rejects_anything_else(value: str):
    assert resolve_date(value, TODAY) is None


def test_today_follows_search_consoles_pacific_day():
    # October is daylight time (UTC-7), January standard time (UTC-8).
    assert pacific_today(datetime(2026, 10, 5, 6, 59, tzinfo=timezone.utc)) == date(
        2026, 10, 4
    )
    assert pacific_today(datetime(2026, 10, 5, 7, 0, tzinfo=timezone.utc)) == TODAY
    assert pacific_today(datetime(2026, 1, 15, 7, 59, tzinfo=timezone.utc)) == date(
        2026, 1, 14
    )
    assert pacific_today(datetime(2026, 1, 15, 8, 0, tzinfo=timezone.utc)) == date(
        2026, 1, 15
    )


def test_resolve_dates_counts_back_from_the_pacific_day(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(_search_console_inputs, "pacific_today", lambda: TODAY)
    assert resolve_dates("30daysAgo", "3daysAgo", "block", "id") == (
        date(2026, 9, 5),
        date(2026, 10, 2),
    )
    assert resolve_dates("2026-10-05", "today", "block", "id") == (TODAY, TODAY)


def test_resolve_dates_names_the_input_that_isnt_a_date():
    with pytest.raises(BlockInputError, match="end_date 'last week' isn't a date"):
        resolve_dates("7daysAgo", "last week", "block", "id", today=TODAY)


def test_resolve_dates_rejects_a_start_after_the_end():
    with pytest.raises(
        BlockInputError,
        match=r"start_date \(2026-10-04\) is after end_date \(2026-10-01\)",
    ):
        resolve_dates("yesterday", "2026-10-01", "block", "id", today=TODAY)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("sc-domain:example.com", "sc-domain:example.com"),
        (" SC-Domain:Example.COM ", "sc-domain:example.com"),
        ("https://www.example.com/", "https://www.example.com/"),
        ("https://WWW.Example.com", "https://www.example.com/"),
        ("HTTP://example.com/blog", "http://example.com/blog/"),
        ("https://example.com/Blog/", "https://example.com/Blog/"),
        ("example.com", None),
        ("www.example.com/", None),
        ("ftp://example.com/", None),
    ],
)
def test_as_property_keeps_properties_and_spots_bare_domains(
    value: str, expected: str | None
):
    assert as_property(value) == expected


def test_site_candidates_try_domain_then_https_then_http():
    assert site_candidates("example.com") == [
        "sc-domain:example.com",
        "sc-domain:www.example.com",
        "https://example.com/",
        "https://www.example.com/",
        "http://example.com/",
        "http://www.example.com/",
    ]
    assert site_candidates("www.example.com")[:4] == [
        "sc-domain:www.example.com",
        "sc-domain:example.com",
        "https://www.example.com/",
        "https://example.com/",
    ]


def _sites(*properties: tuple[str, str]) -> list[SearchConsoleSite]:
    return [
        SearchConsoleSite(site_url=url, permission_level=level)
        for url, level in properties
    ]


def test_match_site_prefers_the_domain_property():
    sites = _sites(
        ("https://www.example.com/", "siteOwner"),
        ("sc-domain:example.com", "siteRestrictedUser"),
    )
    assert match_site("www.example.com", sites) == "sc-domain:example.com"


def test_match_site_prefers_https_and_the_form_given():
    sites = _sites(
        ("http://example.com/", "siteOwner"),
        ("https://www.example.com/", "siteOwner"),
        ("https://example.com/", "siteOwner"),
    )
    assert match_site("example.com", sites) == "https://example.com/"
    assert match_site("www.example.com", sites) == "https://www.example.com/"
    assert match_site("example.com", sites[:1]) == "http://example.com/"


def test_match_site_skips_unverified_and_other_properties():
    sites = _sites(
        ("sc-domain:example.com", "siteUnverifiedUser"),
        ("https://example.com/blog/", "siteOwner"),
        ("https://example.org/", "siteOwner"),
        ("https://example.com/", "siteFullUser"),
    )
    assert match_site("example.com", sites) == "https://example.com/"
    assert match_site("example.net", sites) is None


def test_match_site_for_an_inspection_needs_a_property_with_the_url():
    sites = _sites(
        ("https://example.com/", "siteOwner"),
        ("https://www.example.com/", "siteOwner"),
    )
    url = "https://www.example.com/pricing"
    assert match_site("example.com", sites, url) == "https://www.example.com/"
    assert match_site("example.com", sites, "https://shop.example.com/") is None


@pytest.mark.parametrize(
    "site_url, url, expected",
    [
        ("sc-domain:example.com", "https://example.com/", True),
        ("sc-domain:example.com", "http://shop.example.com/a?b=1", True),
        ("sc-domain:example.com", "https://notexample.com/", False),
        ("sc-domain:shop.example.com", "https://example.com/", False),
        ("https://www.example.com/", "https://www.example.com/pricing", True),
        ("https://www.example.com/", "https://WWW.EXAMPLE.COM", True),
        ("https://www.example.com/", "http://www.example.com/pricing", False),
        ("https://www.example.com/", "https://example.com/pricing", False),
        ("https://www.example.com/blog/", "https://www.example.com/blog/post", True),
        ("https://www.example.com/blog/", "https://www.example.com/blogroll", False),
        ("https://Example.com/Blog/", "https://example.com/Blog/post", True),
        ("https://example.com/Blog/", "https://example.com/blog/post", False),
        ("SC-DOMAIN:Example.com", "https://www.example.com/", True),
    ],
)
def test_property_contains(site_url: str, url: str, expected: bool):
    assert property_contains(site_url, url) is expected


async def test_a_property_is_used_without_listing_sites():
    def list_sites() -> list[dict]:
        raise AssertionError("sites.list shouldn't be called for a property")

    site_url = await resolve_site_url("https://www.example.com", list_sites, "b", "i")
    assert site_url == "https://www.example.com/"


async def test_a_bare_domain_lists_sites_once():
    calls: list[str] = []

    def list_sites() -> list[dict]:
        calls.append("sites.list")
        return [{"siteUrl": "sc-domain:example.com", "permissionLevel": "SITE_OWNER"}]

    assert await resolve_site_url(" Example.com/ ", list_sites, "b", "i") == (
        "sc-domain:example.com"
    )
    assert calls == ["sites.list"]


async def test_no_match_lists_the_properties_the_account_can_read():
    entries = [
        {"siteUrl": f"https://site{n}.example.org/", "permissionLevel": "siteOwner"}
        for n in range(12)
    ]
    entries.append(
        {"siteUrl": "sc-domain:example.com", "permissionLevel": "siteUnverifiedUser"}
    )
    with pytest.raises(BlockExecutionError) as caught:
        await resolve_site_url("example.com", lambda: entries, "b", "i")
    message = str(caught.value)
    assert "matches example.com." in message
    assert "https://site0.example.org/, https://site1.example.org/" in message
    assert "https://site9.example.org/ and 2 more." in message
    assert "site10" not in message and "sc-domain:example.com" not in message


async def test_no_match_for_an_inspection_names_the_url():
    entries = [{"siteUrl": "https://example.com/", "permissionLevel": "siteOwner"}]
    with pytest.raises(
        BlockExecutionError,
        match="matches example.com that contains https://www.example.com/a",
    ):
        await resolve_site_url(
            "example.com",
            lambda: entries,
            "b",
            "i",
            inspection_url="https://www.example.com/a",
        )


async def test_an_account_without_readable_properties_is_told_so():
    entries = [
        {"siteUrl": "sc-domain:example.com", "permissionLevel": "siteUnverifiedUser"}
    ]
    with pytest.raises(BlockExecutionError, match="can't read any Search Console"):
        await resolve_site_url("example.com", lambda: entries, "b", "i")


@pytest.mark.parametrize(
    "value, message",
    [
        ("   ", "Give the Search Console property"),
        ("example.com/blog", "isn't a Search Console property or a domain"),
        ("localhost", "isn't a Search Console property or a domain"),
        ("my site.com", "isn't a Search Console property or a domain"),
    ],
)
async def test_resolve_site_url_rejects_what_isnt_a_property_or_domain(
    value: str, message: str
):
    with pytest.raises(BlockInputError, match=message):
        await resolve_site_url(value, lambda: [], "b", "i")


async def test_a_sites_list_error_gets_a_readable_message():
    content = json.dumps(
        {
            "error": {
                "code": 403,
                "message": "Request had insufficient authentication scopes.",
                "status": "PERMISSION_DENIED",
            }
        }
    ).encode()

    def list_sites() -> list[dict]:
        raise HttpError(httplib2.Response({"status": 403}), content)

    with pytest.raises(BlockExecutionError, match="Reconnect Google"):
        await resolve_site_url("example.com", list_sites, "b", "i")


@pytest.mark.parametrize(
    "value", ["", "www.example.com/page", "ftp://example.com/a", "https://"]
)
def test_require_page_url_wants_a_full_web_url(value: str):
    with pytest.raises(BlockInputError, match="full URL of the page"):
        require_page_url(value, "b", "i")


def test_require_page_url_trims_the_url():
    url = " https://www.example.com/a?b=1 "
    assert require_page_url(url, "b", "i") == url.strip()
