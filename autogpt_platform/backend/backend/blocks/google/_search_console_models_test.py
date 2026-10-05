"""Unit tests for reading Search Console API responses into the blocks' models."""

import pytest

from backend.blocks.google._search_console_models import (
    SearchConsoleRow,
    SearchConsoleSite,
    SearchConsoleSitemap,
    SearchConsoleSitemapContent,
    camel_enum,
    inspection_outputs,
    to_row,
    to_site,
    to_sitemap,
)
from backend.blocks.google._search_console_testdata import TEST_INSPECTION_RESULT


@pytest.mark.parametrize(
    "value, expected",
    [
        ("SITE_OWNER", "siteOwner"),
        ("SITE_UNVERIFIED_USER", "siteUnverifiedUser"),
        ("siteFullUser", "siteFullUser"),
        ("RSS_FEED", "rssFeed"),
        ("WEB", "web"),
        ("androidApp", "androidApp"),
        ("", ""),
    ],
)
def test_camel_enum_reads_both_spellings_of_googles_enums(value: str, expected: str):
    assert camel_enum(value) == expected


def test_to_site():
    assert to_site(
        {"siteUrl": "sc-domain:example.com", "permissionLevel": "SITE_FULL_USER"}
    ) == SearchConsoleSite(
        site_url="sc-domain:example.com", permission_level="siteFullUser"
    )


def test_to_row_fills_the_requested_dimensions_in_order():
    row = to_row(
        {
            "keys": ["usa", "2026-09-30", "MOBILE"],
            "clicks": 12.0,
            "impressions": 340.0,
            "ctr": 0.0352,
            "position": 3.25,
        },
        ["country", "date", "device"],
    )
    assert row == SearchConsoleRow(
        country="usa",
        date="2026-09-30",
        device="MOBILE",
        clicks=12,
        impressions=340,
        ctr=0.0352,
        position=3.25,
    )
    assert isinstance(row.clicks, int) and isinstance(row.impressions, int)


def test_to_row_without_dimensions_is_a_totals_row():
    row = to_row({"clicks": 1520, "impressions": 48210, "ctr": 0.0315}, [])
    assert row == SearchConsoleRow(
        clicks=1520, impressions=48210, ctr=0.0315, position=None
    )


def test_to_row_reads_missing_metrics_as_zero_and_no_position():
    row = to_row({"keys": ["https://www.example.com/a"], "position": 0}, ["page"])
    assert row == SearchConsoleRow(
        page="https://www.example.com/a", clicks=0, impressions=0, ctr=0.0
    )
    assert row.position is None


def test_to_row_maps_search_appearance():
    row = to_row({"keys": ["VIDEO"], "clicks": 3}, ["searchAppearance"])
    assert row.search_appearance == "VIDEO"


def test_to_sitemap_reads_counts_and_drops_indexed():
    sitemap = to_sitemap(
        {
            "path": "https://www.example.com/sitemap_index.xml",
            "lastSubmitted": "2026-08-01T10:15:00.000Z",
            "isPending": True,
            "isSitemapsIndex": True,
            "type": "SITEMAP",
            "lastDownloaded": "2026-10-03T22:41:07.512Z",
            "warnings": "3",
            "errors": "1",
            "contents": [
                {"type": "WEB", "submitted": "1520", "indexed": "0"},
                {"type": "video", "submitted": "12"},
            ],
        }
    )
    assert sitemap == SearchConsoleSitemap(
        path="https://www.example.com/sitemap_index.xml",
        type="sitemap",
        last_submitted="2026-08-01T10:15:00.000Z",
        last_downloaded="2026-10-03T22:41:07.512Z",
        is_pending=True,
        is_sitemaps_index=True,
        warnings=3,
        errors=1,
        contents=[
            SearchConsoleSitemapContent(type="web", submitted=1520),
            SearchConsoleSitemapContent(type="video", submitted=12),
        ],
    )
    assert "indexed" not in sitemap.model_dump_json()


def test_to_sitemap_handles_missing_fields():
    assert to_sitemap({"path": "https://www.example.com/feed.xml"}) == (
        SearchConsoleSitemap(path="https://www.example.com/feed.xml")
    )


def test_inspection_outputs_flatten_an_indexed_page():
    link = TEST_INSPECTION_RESULT["inspectionResultLink"]
    url = "https://www.example.com/pricing"
    assert list(inspection_outputs(TEST_INSPECTION_RESULT)) == [
        ("verdict", "PASS"),
        ("coverage_state", "Submitted and indexed"),
        ("indexing_state", "INDEXING_ALLOWED"),
        ("robots_txt_state", "ALLOWED"),
        ("page_fetch_state", "SUCCESSFUL"),
        ("last_crawl_time", "2026-09-28T08:39:51Z"),
        ("crawled_as", "MOBILE"),
        ("google_canonical", url),
        ("user_canonical", url),
        ("sitemaps", ["https://www.example.com/sitemap.xml"]),
        (
            "referring_urls",
            ["https://www.example.com/", "https://www.example.com/blog/"],
        ),
        ("rich_results_verdict", "PASS"),
        ("rich_result_types", ["Breadcrumbs", "FAQ"]),
        ("inspection_result_link", link),
    ]


def test_inspection_outputs_skip_what_google_leaves_out():
    result = {
        "indexStatusResult": {
            "verdict": "NEUTRAL",
            "coverageState": "Discovered - currently not indexed",
            "robotsTxtState": "ROBOTS_TXT_STATE_UNSPECIFIED",
            "indexingState": "INDEXING_STATE_UNSPECIFIED",
            "pageFetchState": "PAGE_FETCH_STATE_UNSPECIFIED",
        },
        "mobileUsabilityResult": {"verdict": "VERDICT_UNSPECIFIED"},
    }
    assert list(inspection_outputs(result)) == [
        ("verdict", "NEUTRAL"),
        ("coverage_state", "Discovered - currently not indexed"),
        ("indexing_state", "INDEXING_STATE_UNSPECIFIED"),
        ("robots_txt_state", "ROBOTS_TXT_STATE_UNSPECIFIED"),
        ("page_fetch_state", "PAGE_FETCH_STATE_UNSPECIFIED"),
        ("sitemaps", []),
        ("referring_urls", []),
        ("rich_result_types", []),
    ]


def test_inspection_outputs_list_each_rich_result_type_once():
    result = {
        "richResultsResult": {
            "verdict": "FAIL",
            "detectedItems": [
                {"richResultType": "Product snippets", "items": [{"name": "Shoe"}]},
                {"items": [{"name": "Unnamed item"}]},
                {"richResultType": "Product snippets", "items": [{"name": "Boot"}]},
            ],
        }
    }
    outputs = dict(inspection_outputs(result))
    assert outputs["rich_results_verdict"] == "FAIL"
    assert outputs["rich_result_types"] == ["Product snippets"]
    assert "verdict" not in outputs
