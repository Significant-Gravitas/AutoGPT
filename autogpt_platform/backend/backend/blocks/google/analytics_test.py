"""Unit tests for Google Analytics List Properties and List Dimensions and Metrics.

The blocks' own test_input/test_mock cases mock the API calls away. These run
the blocks against real Admin and Data API clients built from the bundled
discovery documents, with only the HTTP layer replaced by canned responses.
"""

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from googleapiclient.discovery import build
from googleapiclient.http import HttpMockSequence

from backend.blocks.google import analytics
from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google.analytics import (
    GoogleAnalyticsDimension,
    GoogleAnalyticsListDimensionsAndMetricsBlock,
    GoogleAnalyticsListPropertiesBlock,
    GoogleAnalyticsMetric,
    GoogleAnalyticsProperty,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError

METADATA = {
    "name": "properties/123456789/metadata",
    "dimensions": [
        {
            "apiName": "country",
            "uiName": "Country",
            "description": "The country from which the user activity originated.",
            "category": "Geography",
        },
        {
            "apiName": "customUser:plan",
            "uiName": "Plan",
            "category": "Custom",
            "customDefinition": True,
        },
    ],
    "metrics": [
        {
            "apiName": "averageSessionDuration",
            "uiName": "Average session duration",
            "description": "The average duration (in seconds) of users' sessions.",
            "type": "TYPE_SECONDS",
            "category": "Session",
        },
        {
            "apiName": "customEvent:order_value",
            "uiName": "Order value",
            "type": "TYPE_CURRENCY",
            "category": "Custom",
            "customDefinition": True,
        },
    ],
}


@pytest.fixture
def google_api(monkeypatch: pytest.MonkeyPatch):
    """Serve canned responses to one of the analytics module's API clients."""

    def install(
        builder: str, api: str, *responses: tuple[int, dict[str, Any]]
    ) -> HttpMockSequence:
        http = HttpMockSequence(
            [({"status": str(status)}, json.dumps(body)) for status, body in responses]
        )
        service = build(api, "v1beta", http=http, cache_discovery=False)
        monkeypatch.setattr(analytics, builder, lambda credentials: service)
        return http

    return install


async def _run(block, **inputs) -> list[tuple[str, Any]]:
    input_data = block.input_schema.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **inputs}
    )
    return [out async for out in block.run(input_data, credentials=TEST_CREDENTIALS)]


@pytest.mark.asyncio
async def test_list_properties_follows_every_page(google_api):
    http = google_api(
        "build_admin_service",
        "analyticsadmin",
        (
            200,
            {
                "accountSummaries": [
                    {
                        "name": "accountSummaries/100",
                        "account": "accounts/100",
                        "displayName": "Acme",
                        "propertySummaries": [
                            {
                                "property": "properties/111",
                                "displayName": "acme.com",
                                "propertyType": "PROPERTY_TYPE_ORDINARY",
                                "parent": "accounts/100",
                            },
                            {
                                "property": "properties/112",
                                "displayName": "Acme roll-up",
                                "propertyType": "PROPERTY_TYPE_ROLLUP",
                                "parent": "accounts/100",
                            },
                        ],
                    },
                    {"name": "accountSummaries/200", "account": "accounts/200"},
                ],
                "nextPageToken": "page-2",
            },
        ),
        (
            200,
            {
                "accountSummaries": [
                    {
                        "name": "accountSummaries/300",
                        "account": "accounts/300",
                        "displayName": "Client Co",
                        "propertySummaries": [
                            {
                                "property": "properties/333",
                                "propertyType": "PROPERTY_TYPE_SUBPROPERTY",
                                "parent": "properties/111",
                            },
                            {"property": "properties/334"},
                        ],
                    }
                ]
            },
        ),
    )
    outputs = await _run(GoogleAnalyticsListPropertiesBlock())

    queries = [parse_qs(urlparse(uri).query) for uri, *_ in http.request_sequence]
    assert all(
        urlparse(uri).path == "/v1beta/accountSummaries" and method == "GET"
        for uri, method, *_ in http.request_sequence
    )
    assert queries == [
        {"pageSize": ["200"], "alt": ["json"]},
        {"pageSize": ["200"], "pageToken": ["page-2"], "alt": ["json"]},
    ]
    properties = [
        GoogleAnalyticsProperty(
            property_id="111",
            name="properties/111",
            display_name="acme.com",
            property_type="ordinary",
            account_id="100",
            account_name="Acme",
        ),
        GoogleAnalyticsProperty(
            property_id="112",
            name="properties/112",
            display_name="Acme roll-up",
            property_type="rollup",
            account_id="100",
            account_name="Acme",
        ),
        GoogleAnalyticsProperty(
            property_id="333",
            name="properties/333",
            property_type="subproperty",
            account_id="300",
            account_name="Client Co",
        ),
        GoogleAnalyticsProperty(
            property_id="334",
            name="properties/334",
            property_type="unspecified",
            account_id="300",
            account_name="Client Co",
        ),
    ]
    assert outputs == [
        ("properties", properties),
        *[("property", item) for item in properties],
    ]


@pytest.mark.asyncio
async def test_list_properties_with_no_access_to_any_account(google_api):
    google_api("build_admin_service", "analyticsadmin", (200, {}))
    assert await _run(GoogleAnalyticsListPropertiesBlock()) == [("properties", [])]


@pytest.mark.asyncio
async def test_list_properties_names_the_admin_api_when_it_is_off(google_api):
    google_api(
        "build_admin_service",
        "analyticsadmin",
        (
            403,
            {
                "error": {
                    "code": 403,
                    "message": "Google Analytics Admin API has not been used in "
                    "project 42 before or it is disabled.",
                    "status": "PERMISSION_DENIED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                            "reason": "SERVICE_DISABLED",
                            "domain": "googleapis.com",
                        }
                    ],
                }
            },
        ),
    )
    with pytest.raises(
        BlockExecutionError,
        match="The Google Analytics Admin API isn't enabled for this AutoGPT instance",
    ):
        await _run(GoogleAnalyticsListPropertiesBlock())


@pytest.mark.asyncio
async def test_list_dimensions_and_metrics_lists_only_custom_ones_by_default(
    google_api,
):
    http = google_api("build_data_service", "analyticsdata", (200, METADATA))
    outputs = await _run(
        GoogleAnalyticsListDimensionsAndMetricsBlock(), property_id="123456789"
    )
    [(uri, method, body, _)] = http.request_sequence
    assert (method, uri, body) == (
        "GET",
        "https://analyticsdata.googleapis.com/v1beta/properties/123456789/metadata"
        "?alt=json",
        None,
    )
    assert outputs == [
        (
            "dimensions",
            [
                GoogleAnalyticsDimension(
                    api_name="customUser:plan",
                    ui_name="Plan",
                    category="Custom",
                    custom=True,
                )
            ],
        ),
        (
            "metrics",
            [
                GoogleAnalyticsMetric(
                    api_name="customEvent:order_value",
                    ui_name="Order value",
                    category="Custom",
                    custom=True,
                    type="currency",
                )
            ],
        ),
    ]


@pytest.mark.asyncio
async def test_list_dimensions_and_metrics_can_include_the_standard_ones(google_api):
    google_api("build_data_service", "analyticsdata", (200, METADATA))
    outputs = dict(
        await _run(
            GoogleAnalyticsListDimensionsAndMetricsBlock(),
            property_id="properties/123456789",
            custom_only=False,
        )
    )
    assert [item.api_name for item in outputs["dimensions"]] == [
        "country",
        "customUser:plan",
    ]
    assert outputs["metrics"][0] == GoogleAnalyticsMetric(
        api_name="averageSessionDuration",
        ui_name="Average session duration",
        description="The average duration (in seconds) of users' sessions.",
        category="Session",
        custom=False,
        type="seconds",
    )


@pytest.mark.asyncio
async def test_list_dimensions_and_metrics_checks_the_property_id(google_api):
    http = google_api("build_data_service", "analyticsdata")
    with pytest.raises(BlockInputError, match="Universal Analytics"):
        await _run(
            GoogleAnalyticsListDimensionsAndMetricsBlock(), property_id="UA-99-1"
        )
    assert http.request_sequence == []


@pytest.mark.asyncio
async def test_list_dimensions_and_metrics_explains_a_missing_property(google_api):
    google_api(
        "build_data_service",
        "analyticsdata",
        (
            403,
            {
                "error": {
                    "code": 403,
                    "message": "User does not have sufficient permissions for this "
                    "property.",
                    "status": "PERMISSION_DENIED",
                }
            },
        ),
    )
    with pytest.raises(BlockExecutionError, match="property 987"):
        await _run(GoogleAnalyticsListDimensionsAndMetricsBlock(), property_id="987")
