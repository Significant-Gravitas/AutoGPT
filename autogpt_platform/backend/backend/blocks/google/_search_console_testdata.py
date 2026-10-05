"""Canned Search Console API responses for the blocks' self-tests and unit tests.

They follow the examples in Google's API reference and its URL Inspection API
announcement.
"""

from typing import Any

TEST_SITE_ENTRIES: list[dict[str, Any]] = [
    {"siteUrl": "sc-domain:example.com", "permissionLevel": "siteOwner"},
    {"siteUrl": "https://www.example.com/", "permissionLevel": "siteFullUser"},
    {"siteUrl": "https://shop.example.org/", "permissionLevel": "siteUnverifiedUser"},
]

TEST_ANALYTICS_RESPONSE: dict[str, Any] = {
    "rows": [
        {
            "keys": ["running shoes", "https://www.example.com/shoes/"],
            "clicks": 120,
            "impressions": 3400,
            "ctr": 0.0353,
            "position": 4.2,
        },
        {
            "keys": ["trail shoes", "https://www.example.com/trail/"],
            "clicks": 45,
            "impressions": 2100,
            "ctr": 0.0214,
            "position": 7.8,
        },
    ],
    "responseAggregationType": "byPage",
}

TEST_INSPECTED_URL = "https://www.example.com/pricing"

TEST_INSPECTION_RESULT: dict[str, Any] = {
    "inspectionResultLink": (
        "https://search.google.com/search-console/inspect"
        "?resource_id=sc-domain:example.com&id=odaUL5Dqq3q8n0EicQzawg"
    ),
    "indexStatusResult": {
        "verdict": "PASS",
        "coverageState": "Submitted and indexed",
        "robotsTxtState": "ALLOWED",
        "indexingState": "INDEXING_ALLOWED",
        "lastCrawlTime": "2026-09-28T08:39:51Z",
        "pageFetchState": "SUCCESSFUL",
        "googleCanonical": TEST_INSPECTED_URL,
        "userCanonical": TEST_INSPECTED_URL,
        "sitemap": ["https://www.example.com/sitemap.xml"],
        "referringUrls": ["https://www.example.com/", "https://www.example.com/blog/"],
        "crawledAs": "MOBILE",
    },
    "mobileUsabilityResult": {"verdict": "PASS"},
    "richResultsResult": {
        "verdict": "PASS",
        "detectedItems": [
            {"richResultType": "Breadcrumbs", "items": [{"name": "Unnamed item"}]},
            {"richResultType": "FAQ", "items": [{"name": "Unnamed item"}]},
        ],
    },
}

TEST_SITEMAPS_RESPONSE: dict[str, Any] = {
    "sitemap": [
        {
            "path": "https://www.example.com/sitemap.xml",
            "lastSubmitted": "2026-08-01T10:15:00.000Z",
            "isPending": False,
            "isSitemapsIndex": False,
            "type": "sitemap",
            "lastDownloaded": "2026-10-03T22:41:07.512Z",
            "warnings": "2",
            "errors": "0",
            "contents": [
                {"type": "web", "submitted": "1520", "indexed": "0"},
                {"type": "image", "submitted": "310", "indexed": "0"},
            ],
        }
    ]
}
