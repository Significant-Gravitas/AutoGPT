"""Unit tests for Google Analytics Run Report.

The block's own test_input/test_mock case mocks the API call away. These run
the block against a real Data API client built from the bundled discovery
document, with only the HTTP layer replaced by canned responses, so they check
the request the block sends, how it reads the report, and how it reports errors.
"""

import json
from typing import Any

import pytest
from googleapiclient.discovery import build
from googleapiclient.http import HttpMockSequence

from backend.blocks.google import analytics_reports
from backend.blocks.google._analytics_api import FIELDS_HINT
from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google.analytics_reports import GoogleAnalyticsRunReportBlock
from backend.util.exceptions import BlockExecutionError, BlockInputError

REPORT_URL = (
    "https://analyticsdata.googleapis.com/v1beta/properties/123456789:runReport"
    "?alt=json"
)


@pytest.fixture
def data_api(monkeypatch: pytest.MonkeyPatch):
    """Serve canned Data API responses to the block's client."""

    def install(*responses: tuple[int, dict[str, Any]]) -> HttpMockSequence:
        http = HttpMockSequence(
            [({"status": str(status)}, json.dumps(body)) for status, body in responses]
        )
        service = build("analyticsdata", "v1beta", http=http, cache_discovery=False)
        monkeypatch.setattr(
            analytics_reports, "build_data_service", lambda credentials: service
        )
        return http

    return install


async def _run(**inputs) -> list[tuple[str, Any]]:
    block = GoogleAnalyticsRunReportBlock()
    input_data = block.input_schema.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **inputs}
    )
    return [out async for out in block.run(input_data, credentials=TEST_CREDENTIALS)]


@pytest.mark.asyncio
async def test_run_report_sends_the_request_and_reads_the_report(data_api):
    http = data_api(
        (
            200,
            {
                "dimensionHeaders": [{"name": "date"}],
                "metricHeaders": [
                    {"name": "activeUsers", "type": "TYPE_INTEGER"},
                    {"name": "averageSessionDuration", "type": "TYPE_SECONDS"},
                ],
                "rows": [
                    {
                        "dimensionValues": [{"value": "20260930"}],
                        "metricValues": [{"value": "310"}, {"value": "74.5"}],
                    },
                    {
                        "dimensionValues": [{"value": "20260929"}],
                        "metricValues": [{"value": "288"}, {"value": "80"}],
                    },
                ],
                "totals": [
                    {
                        "dimensionValues": [{"value": "RESERVED_TOTAL"}],
                        "metricValues": [{"value": "560"}, {"value": "77.1"}],
                    }
                ],
                "rowCount": 7,
                "metadata": {"currencyCode": "GBP", "timeZone": "Europe/London"},
                "kind": "analyticsData#runReport",
            },
        )
    )
    outputs = await _run(
        property_id=" properties/123456789 ",
        metrics=["activeUsers, averageSessionDuration"],
        dimensions=["date"],
        start_date="7DaysAgo",
        end_date="2026-09-30",
        dimension_filters=[
            {"dimension": "pagePath", "match_type": "begins_with", "value": "/blog/"},
            {"dimension": "deviceCategory", "value": "tablet", "exclude": True},
        ],
        order_by="-date",
        limit=2,
    )

    [(uri, method, body, headers)] = http.request_sequence
    assert (method, uri) == ("POST", REPORT_URL)
    assert json.loads(body) == {
        "dateRanges": [{"startDate": "7daysAgo", "endDate": "2026-09-30"}],
        "metrics": [{"name": "activeUsers"}, {"name": "averageSessionDuration"}],
        "dimensions": [{"name": "date"}],
        "dimensionFilter": {
            "andGroup": {
                "expressions": [
                    {
                        "filter": {
                            "fieldName": "pagePath",
                            "stringFilter": {
                                "matchType": "BEGINS_WITH",
                                "value": "/blog/",
                                "caseSensitive": False,
                            },
                        }
                    },
                    {
                        "notExpression": {
                            "filter": {
                                "fieldName": "deviceCategory",
                                "stringFilter": {
                                    "matchType": "EXACT",
                                    "value": "tablet",
                                    "caseSensitive": False,
                                },
                            }
                        }
                    },
                ]
            }
        },
        "orderBys": [{"dimension": {"dimensionName": "date"}, "desc": True}],
        "limit": "2",
        "metricAggregations": ["TOTAL"],
    }
    rows = [
        {"date": "20260930", "activeUsers": 310, "averageSessionDuration": 74.5},
        {"date": "20260929", "activeUsers": 288, "averageSessionDuration": 80.0},
    ]
    assert outputs == [
        ("rows", rows),
        ("row", rows[0]),
        ("row", rows[1]),
        ("totals", {"activeUsers": 560, "averageSessionDuration": 77.1}),
        ("row_count", 7),
        ("time_zone", "Europe/London"),
        ("currency_code", "GBP"),
    ]


@pytest.mark.asyncio
async def test_run_report_uses_the_default_metrics_and_dates(data_api):
    http = data_api((200, {}))
    await _run(property_id="123456789")
    [(uri, _, body, _)] = http.request_sequence
    assert uri == REPORT_URL
    assert json.loads(body) == {
        "dateRanges": [{"startDate": "28daysAgo", "endDate": "yesterday"}],
        "metrics": [{"name": "activeUsers"}, {"name": "sessions"}],
        "limit": "100",
        "metricAggregations": ["TOTAL"],
    }


@pytest.mark.asyncio
async def test_run_report_with_no_data(data_api):
    data_api(
        (
            200,
            {
                "dimensionHeaders": [{"name": "country"}],
                "metricHeaders": [{"name": "sessions", "type": "TYPE_INTEGER"}],
                "totals": [{}],
                "metadata": {"currencyCode": "USD", "timeZone": "America/Chicago"},
                "kind": "analyticsData#runReport",
            },
        )
    )
    outputs = await _run(property_id="123456789", dimensions=["country"])
    assert outputs == [
        ("rows", []),
        ("totals", {}),
        ("row_count", 0),
        ("time_zone", "America/Chicago"),
        ("currency_code", "USD"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inputs, problem",
    [
        ({"property_id": "G-ABC123XYZ"}, "measurement ID"),
        ({"property_id": "UA-1234-1"}, "Universal Analytics"),
        ({"property_id": "123", "start_date": "last month"}, "start_date must be"),
        ({"property_id": "123", "end_date": "2026-13-01"}, "end_date must be"),
        ({"property_id": "123", "metrics": []}, "at least one metric"),
        ({"property_id": "123", "order_by": "-pageViews"}, "Can't sort by 'pageViews'"),
        (
            {
                "property_id": "123",
                "dimension_filters": [{"dimension": "x", "value": ""}],
            },
            "needs a dimension name and a value",
        ),
    ],
)
async def test_run_report_rejects_bad_inputs_before_calling_google(
    data_api, inputs: dict, problem: str
):
    http = data_api()
    with pytest.raises(BlockInputError, match=problem):
        await _run(**inputs)
    assert http.request_sequence == []


@pytest.mark.asyncio
async def test_run_report_explains_a_property_the_account_cant_read(data_api):
    data_api(
        (
            403,
            {
                "error": {
                    "code": 403,
                    "message": "User does not have sufficient permissions for this "
                    "property. To learn more about Property ID, see https://"
                    "developers.google.com/analytics/devguides/reporting/data/v1/"
                    "property-id.",
                    "status": "PERMISSION_DENIED",
                }
            },
        )
    )
    with pytest.raises(BlockExecutionError, match="at least Viewer access") as raised:
        await _run(property_id="123456789")
    assert "property 123456789" in str(raised.value)


@pytest.mark.asyncio
async def test_run_report_passes_an_invalid_name_through(data_api):
    reason = (
        "Field pageViews is not a valid metric. For a list of valid dimensions and "
        "metrics, see https://developers.google.com/analytics/devguides/reporting/"
        "data/v1/api-schema"
    )
    data_api(
        (400, {"error": {"code": 400, "message": reason, "status": "INVALID_ARGUMENT"}})
    )
    with pytest.raises(BlockExecutionError) as raised:
        await _run(property_id="123456789", metrics=["pageViews"])
    assert str(raised.value) == (
        f"Google Analytics rejected the request: {reason} {FIELDS_HINT}"
    )
