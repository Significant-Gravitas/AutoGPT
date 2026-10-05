"""Unit tests for Google Analytics Run Realtime Report.

Like analytics_reports_test.py, these run the block against a real Data API
client with canned HTTP responses.
"""

import json
from typing import Any

import pytest
from googleapiclient.discovery import build
from googleapiclient.http import HttpMockSequence

from backend.blocks.google import analytics_realtime
from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google.analytics_realtime import (
    REALTIME_HINT,
    GoogleAnalyticsRunRealtimeReportBlock,
    realtime_hint,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError

REALTIME_URL = (
    "https://analyticsdata.googleapis.com/v1beta/properties/123456789"
    ":runRealtimeReport?alt=json"
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
            analytics_realtime, "build_data_service", lambda credentials: service
        )
        return http

    return install


async def _run(**inputs) -> list[tuple[str, Any]]:
    block = GoogleAnalyticsRunRealtimeReportBlock()
    input_data = block.input_schema.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **inputs}
    )
    return [out async for out in block.run(input_data, credentials=TEST_CREDENTIALS)]


@pytest.mark.asyncio
async def test_realtime_report_asks_for_the_last_minutes(data_api):
    http = data_api(
        (
            200,
            {
                "dimensionHeaders": [{"name": "unifiedScreenName"}],
                "metricHeaders": [
                    {"name": "activeUsers", "type": "TYPE_INTEGER"},
                    {"name": "screenPageViews", "type": "TYPE_INTEGER"},
                ],
                "rows": [
                    {
                        "dimensionValues": [{"value": "Pricing"}],
                        "metricValues": [{"value": "12"}, {"value": "30"}],
                    }
                ],
                "totals": [
                    {
                        "dimensionValues": [{"value": "RESERVED_TOTAL"}],
                        "metricValues": [{"value": "12"}, {"value": "30"}],
                    }
                ],
                "rowCount": 1,
                "kind": "analyticsData#runRealtimeReport",
            },
        )
    )
    outputs = await _run(
        property_id="123456789",
        metrics=["activeUsers", "screenPageViews"],
        dimensions=["unifiedScreenName"],
        minutes_ago=10,
        dimension_filters=[
            {
                "dimension": "unifiedScreenName",
                "match_type": "contains",
                "value": "Pric",
            }
        ],
        order_by="-screenPageViews",
        limit=5,
    )

    [(uri, method, body, _)] = http.request_sequence
    assert (method, uri) == ("POST", REALTIME_URL)
    assert json.loads(body) == {
        "minuteRanges": [{"startMinutesAgo": 10, "endMinutesAgo": 0}],
        "metrics": [{"name": "activeUsers"}, {"name": "screenPageViews"}],
        "dimensions": [{"name": "unifiedScreenName"}],
        "dimensionFilter": {
            "filter": {
                "fieldName": "unifiedScreenName",
                "stringFilter": {
                    "matchType": "CONTAINS",
                    "value": "Pric",
                    "caseSensitive": False,
                },
            }
        },
        "orderBys": [{"metric": {"metricName": "screenPageViews"}, "desc": True}],
        "limit": "5",
        "metricAggregations": ["TOTAL"],
    }
    row = {"unifiedScreenName": "Pricing", "activeUsers": 12, "screenPageViews": 30}
    assert outputs == [
        ("rows", [row]),
        ("row", row),
        ("totals", {"activeUsers": 12, "screenPageViews": 30}),
        ("row_count", 1),
    ]


@pytest.mark.asyncio
async def test_realtime_report_defaults_to_active_users_in_the_last_30_minutes(
    data_api,
):
    http = data_api((200, {}))
    outputs = await _run(property_id="properties/123456789")
    [(_, _, body, _)] = http.request_sequence
    assert json.loads(body) == {
        "minuteRanges": [{"startMinutesAgo": 29, "endMinutesAgo": 0}],
        "metrics": [{"name": "activeUsers"}],
        "limit": "100",
        "metricAggregations": ["TOTAL"],
    }
    assert outputs == [("rows", []), ("totals", {}), ("row_count", 0)]


@pytest.mark.parametrize("minutes_ago", [-1, 60])
def test_minutes_ago_is_bounded_by_what_360_properties_allow(minutes_ago: int):
    schema = GoogleAnalyticsRunRealtimeReportBlock.Input.jsonschema()
    error = GoogleAnalyticsRunRealtimeReportBlock.Input.validate_data(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "property_id": "1",
            "minutes_ago": minutes_ago,
        }
    )
    assert schema["properties"]["minutes_ago"]["maximum"] == 59
    assert error is not None


@pytest.mark.asyncio
async def test_realtime_report_rejects_a_measurement_id(data_api):
    http = data_api()
    with pytest.raises(BlockInputError, match="measurement ID"):
        await _run(property_id="G-ABC123XYZ")
    assert http.request_sequence == []


@pytest.mark.asyncio
@pytest.mark.parametrize("minutes_ago, mentions_360", [(29, False), (45, True)])
async def test_invalid_realtime_requests_point_to_the_realtime_schema(
    data_api, minutes_ago: int, mentions_360: bool
):
    reason = "Field pagePath is not a valid dimension."
    data_api(
        (400, {"error": {"code": 400, "message": reason, "status": "INVALID_ARGUMENT"}})
    )
    with pytest.raises(BlockExecutionError) as raised:
        await _run(
            property_id="123456789", dimensions=["pagePath"], minutes_ago=minutes_ago
        )
    message = str(raised.value)
    assert message.startswith(f"Google Analytics rejected the request: {reason}")
    assert REALTIME_HINT in message
    assert ("Google Analytics 360" in message) is mentions_360


def test_realtime_hint_mentions_360_only_past_the_standard_window():
    assert realtime_hint(0) == REALTIME_HINT
    assert realtime_hint(29) == REALTIME_HINT
    assert realtime_hint(30).endswith(
        "A standard property can look back at most 29 minutes; more needs Google "
        "Analytics 360."
    )
