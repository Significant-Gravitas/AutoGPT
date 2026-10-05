"""Unit tests for the Search Console request body, its match with Google's
discovery document, and the error messages."""

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, get_args

import googleapiclient
import httplib2
import pytest
from googleapiclient.errors import HttpError

from backend.blocks.google._search_console_api import (
    build_query_body,
    search_console_error,
)
from backend.blocks.google._search_console_models import (
    Dimension,
    SearchConsoleFilter,
    SearchType,
)

DISCOVERY = json.loads(
    (
        Path(googleapiclient.__file__).parent
        / "discovery_cache/documents/searchconsole.v1.json"
    ).read_text(encoding="utf-8")
)


def _body(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "start_date": date(2026, 9, 5),
        "end_date": date(2026, 10, 2),
        "dimensions": ["query"],
        "search_type": "web",
        "filters": [],
        "row_limit": 100,
        "start_row": 0,
        "include_fresh_data": False,
        **overrides,
    }
    return build_query_body(**fields)


def test_query_body_defaults():
    assert _body() == {
        "startDate": "2026-09-05",
        "endDate": "2026-10-02",
        "dimensions": ["query"],
        "type": "web",
        "dataState": "final",
        "rowLimit": 100,
        "startRow": 0,
    }


def test_query_body_puts_every_filter_in_one_and_group():
    body = _body(
        dimensions=["page", "device"],
        search_type="discover",
        include_fresh_data=True,
        row_limit=25000,
        start_row=25000,
        filters=[
            SearchConsoleFilter(
                dimension="page", operator="includingRegex", expression="/blog/.*"
            ),
            SearchConsoleFilter(
                dimension="device", operator="equals", expression="MOBILE"
            ),
        ],
    )
    assert body["dimensionFilterGroups"] == [
        {
            "groupType": "and",
            "filters": [
                {
                    "dimension": "page",
                    "operator": "includingRegex",
                    "expression": "/blog/.*",
                },
                {"dimension": "device", "operator": "equals", "expression": "MOBILE"},
            ],
        }
    ]
    assert body["dataState"] == "all"
    assert (body["type"], body["rowLimit"], body["startRow"]) == (
        "discover",
        25000,
        25000,
    )


def _upper_snake(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).upper()


def test_query_body_fields_and_values_match_the_discovery_document():
    """The API reference documents camelCase values and the discovery document
    lists the same enums in upper case, so compare them that way."""
    schemas = DISCOVERY["schemas"]
    request = schemas["SearchAnalyticsQueryRequest"]["properties"]
    group = schemas["ApiDimensionFilterGroup"]["properties"]
    dimension_filter = schemas["ApiDimensionFilter"]["properties"]
    body = _body(
        filters=[
            SearchConsoleFilter(dimension="query", operator="contains", expression="x")
        ]
    )
    assert set(body) <= set(request)
    assert set(body["dimensionFilterGroups"][0]) <= set(group)
    assert set(body["dimensionFilterGroups"][0]["filters"][0]) <= set(dimension_filter)
    assert _upper_snake(body["dataState"]) in request["dataState"]["enum"]
    assert _upper_snake(_body(include_fresh_data=True)["dataState"]) in (
        request["dataState"]["enum"]
    )
    assert _upper_snake(body["dimensionFilterGroups"][0]["groupType"]) in (
        group["groupType"]["enum"]
    )
    assert {_upper_snake(v) for v in get_args(Dimension)} <= set(
        request["dimensions"]["items"]["enum"]
    )
    assert {_upper_snake(v) for v in get_args(SearchType)} == set(
        request["type"]["enum"]
    )
    filter_fields = SearchConsoleFilter.model_fields
    for name in ("dimension", "operator"):
        values = get_args(filter_fields[name].annotation)
        assert {_upper_snake(v) for v in values} == set(dimension_filter[name]["enum"])


def test_every_method_the_blocks_call_accepts_the_read_only_scope():
    resources = DISCOVERY["resources"]
    methods = [
        resources["sites"]["methods"]["list"],
        resources["searchanalytics"]["methods"]["query"],
        resources["sitemaps"]["methods"]["list"],
        resources["urlInspection"]["resources"]["index"]["methods"]["inspect"],
    ]
    for method in methods:
        assert "https://www.googleapis.com/auth/webmasters.readonly" in method["scopes"]


def _http_error(
    status: int,
    message: str,
    *,
    api_status: str = "",
    details: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
) -> HttpError:
    error: dict[str, Any] = {"code": status, "message": message}
    if api_status:
        error["status"] = api_status
    if details:
        error["details"] = [
            {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}
            for reason in details
        ]
    if errors:
        error["errors"] = [{"domain": "global", "reason": reason} for reason in errors]
    content = json.dumps({"error": error}).encode()
    return HttpError(httplib2.Response({"status": status}), content)


SITE = "https://www.example.com/"
PAGE = "https://www.example.com/pricing"


@pytest.mark.parametrize(
    "error, context, expected",
    [
        (
            _http_error(
                403,
                "Google Search Console API has not been used in project 123 before "
                "or it is disabled.",
                api_status="PERMISSION_DENIED",
                details=("SERVICE_DISABLED",),
            ),
            {},
            "The Google Search Console API isn't enabled for this AutoGPT instance",
        ),
        (
            _http_error(403, "Access Not Configured.", errors=("accessNotConfigured",)),
            {},
            "Ask your AutoGPT administrator to enable it",
        ),
        (
            _http_error(
                403,
                "Request had insufficient authentication scopes.",
                api_status="PERMISSION_DENIED",
                details=("ACCESS_TOKEN_SCOPE_INSUFFICIENT",),
                errors=("insufficientPermissions",),
            ),
            {"site_url": SITE},
            "hasn't granted the Search Console access this block needs. Reconnect",
        ),
        (
            _http_error(
                403,
                f"User does not have sufficient permission for site '{SITE}'. See "
                "also: https://support.google.com/webmasters/answer/2451999.",
                api_status="PERMISSION_DENIED",
                errors=("forbidden",),
            ),
            {"site_url": SITE},
            f"can't read the Search Console property {SITE}. Use Google Search "
            "Console List Sites",
        ),
        (
            _http_error(
                403,
                "You do not own this site, or the inspected URL is not part of this "
                "property.",
                api_status="PERMISSION_DENIED",
            ),
            {"site_url": SITE, "inspection_url": PAGE},
            f"can't read the Search Console property {SITE}, or {PAGE} isn't part of it",
        ),
        (
            _http_error(
                429,
                "Quota exceeded for quota metric 'Search analytics requests'.",
                api_status="RESOURCE_EXHAUSTED",
                details=("RATE_LIMIT_EXCEEDED",),
            ),
            {"site_url": SITE},
            "Search Console's quota for this property is used up for now. Try again",
        ),
        (
            _http_error(429, "Quota exceeded.", api_status="RESOURCE_EXHAUSTED"),
            {"site_url": SITE, "inspection_url": PAGE},
            "2,000 inspections a day and 600 a minute per property",
        ),
        (
            _http_error(403, "Rate Limit Exceeded", errors=("rateLimitExceeded",)),
            {"site_url": SITE},
            "quota for this property is used up",
        ),
        (
            _http_error(
                400,
                "Invalid value at 'dimensions[0]'.",
                api_status="INVALID_ARGUMENT",
            ),
            {"site_url": SITE},
            "Google Search Console rejected the request: Invalid value at "
            "'dimensions[0]'.",
        ),
        (
            _http_error(500, "Backend Error", errors=("backendError",)),
            {},
            "Google Search Console API error 500: Backend Error",
        ),
    ],
)
def test_search_console_error_messages(
    error: HttpError, context: dict[str, str], expected: str
):
    mapped = search_console_error(error, "block", "block-id", **context)
    assert expected in str(mapped)
    assert (mapped.block_name, mapped.block_id) == ("block", "block-id")


def test_search_console_error_reads_a_body_that_isnt_json():
    response = httplib2.Response({"status": 502})
    response.reason = "Bad Gateway"
    error = HttpError(response, b"<html><body>Bad Gateway</body></html>")
    assert str(search_console_error(error, "block", "id")) == (
        "Google Search Console API error 502: Bad Gateway"
    )
