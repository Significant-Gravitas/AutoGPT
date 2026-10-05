"""Unit tests for the Google Analytics API clients, property IDs and errors.

The clients are built from the discovery documents bundled with
google-api-python-client, so none of these tests touch the network.
"""

import json

import httplib2
import pytest
from googleapiclient.discovery_cache import get_static_doc
from googleapiclient.errors import HttpError

from backend.blocks.google._analytics_api import (
    ADMIN_API,
    ANALYTICS_READONLY_SCOPE,
    DATA_API,
    FIELDS_HINT,
    analytics_error,
    build_admin_service,
    build_data_service,
    property_resource_name,
)
from backend.blocks.google._auth import TEST_CREDENTIALS
from backend.util.exceptions import BlockInputError

DATA_URL = "https://analyticsdata.googleapis.com/v1beta"
ADMIN_URL = "https://analyticsadmin.googleapis.com/v1beta"


def test_clients_build_the_requests_the_blocks_send_without_the_network():
    data = build_data_service(TEST_CREDENTIALS)
    admin = build_admin_service(TEST_CREDENTIALS)
    body = {"metrics": [{"name": "sessions"}], "limit": "10"}

    report = data.properties().runReport(property="properties/1", body=body)
    realtime = data.properties().runRealtimeReport(property="properties/1", body=body)
    metadata = data.properties().getMetadata(name="properties/1/metadata")
    summaries = admin.accountSummaries().list(pageSize=200)

    assert (report.method, report.uri) == (
        "POST",
        f"{DATA_URL}/properties/1:runReport?alt=json",
    )
    assert json.loads(report.body) == body
    assert (realtime.method, realtime.uri) == (
        "POST",
        f"{DATA_URL}/properties/1:runRealtimeReport?alt=json",
    )
    assert json.loads(realtime.body) == body
    assert (metadata.method, metadata.uri) == (
        "GET",
        f"{DATA_URL}/properties/1/metadata?alt=json",
    )
    assert (summaries.method, summaries.uri) == (
        "GET",
        f"{ADMIN_URL}/accountSummaries?pageSize=200&alt=json",
    )


@pytest.mark.parametrize(
    "api, resource, method",
    [
        ("analyticsdata", "properties", "runReport"),
        ("analyticsdata", "properties", "runRealtimeReport"),
        ("analyticsdata", "properties", "getMetadata"),
        ("analyticsadmin", "accountSummaries", "list"),
    ],
)
def test_every_method_the_blocks_call_accepts_the_read_only_scope(
    api: str, resource: str, method: str
):
    document = json.loads(get_static_doc(api, "v1beta") or "{}")
    scopes = document["resources"][resource]["methods"][method]["scopes"]
    assert ANALYTICS_READONLY_SCOPE in scopes


@pytest.mark.parametrize(
    "value",
    ["123456789", " 123456789 ", "properties/123456789", " properties/123456789"],
)
def test_property_resource_name_accepts_ids_and_resource_names(value: str):
    assert property_resource_name(value, "block", "id") == "properties/123456789"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("G-ABC123XYZ", "is a measurement ID"),
        ("g-abc123xyz", "is a measurement ID"),
        ("properties/G-ABC123XYZ", "G-ABC123XYZ is a measurement ID"),
        ("UA-12345678-1", "Universal Analytics was shut down"),
        ("", "Enter a Google Analytics 4 property ID"),
        ("example.com", "Enter a Google Analytics 4 property ID"),
        ("123-456", "Enter a Google Analytics 4 property ID"),
        ("properties/", "Enter a Google Analytics 4 property ID"),
        ("accounts/123", "Enter a Google Analytics 4 property ID"),
    ],
)
def test_property_resource_name_says_where_to_find_the_id(value: str, expected: str):
    with pytest.raises(BlockInputError, match=expected) as raised:
        property_resource_name(value, "block", "id")
    assert "Admin > Property details" in str(raised.value)
    assert "Google Analytics List Properties" in str(raised.value)


def test_service_disabled_is_an_operator_problem():
    error = _http_error(
        403,
        "Google Analytics Data API has not been used in project 123456 before or "
        "it is disabled. Enable it by visiting https://console.developers.google.com"
        "/apis/api/analyticsdata.googleapis.com/overview?project=123456 then retry.",
        "PERMISSION_DENIED",
        "SERVICE_DISABLED",
    )
    message = str(analytics_error(error, "block", "id", api=DATA_API))
    assert message.startswith(
        "The Google Analytics Data API isn't enabled for this AutoGPT instance."
    )
    assert "administrator" in message
    assert "has not been used in project 123456" in message


def test_service_disabled_is_recognised_by_its_message_too():
    error = _http_error(
        403,
        "Google Analytics Admin API has not been used in project 9 before or it is "
        "disabled.",
    )
    message = str(analytics_error(error, "block", "id", api=ADMIN_API))
    assert "The Google Analytics Admin API isn't enabled" in message


def test_missing_scope_asks_the_user_to_reconnect():
    error = _http_error(
        403,
        "Request had insufficient authentication scopes.",
        "PERMISSION_DENIED",
        "ACCESS_TOKEN_SCOPE_INSUFFICIENT",
    )
    message = str(
        analytics_error(
            error, "block", "id", api=DATA_API, property_name="properties/1"
        )
    )
    assert message == (
        "The connected Google account hasn't granted the Google Analytics access "
        "this block needs. Reconnect Google and approve Google Analytics access."
    )


def test_no_access_to_the_property_names_the_property_and_the_fix():
    error = _http_error(
        403,
        "User does not have sufficient permissions for this property. To learn more "
        "about Property ID, see https://developers.google.com/analytics/devguides/"
        "reporting/data/v1/property-id.",
        "PERMISSION_DENIED",
    )
    message = str(
        analytics_error(
            error, "block", "id", api=DATA_API, property_name="properties/123456789"
        )
    )
    assert "can't read Google Analytics property 123456789" in message
    assert "at least Viewer access" in message
    assert "Google Analytics List Properties" in message
    assert "User does not have sufficient permissions" in message


def test_other_permission_errors_without_a_property_pass_google_through():
    error = _http_error(403, "The caller does not have permission", "PERMISSION_DENIED")
    message = str(analytics_error(error, "block", "id", api=ADMIN_API))
    assert message == (
        "Google Analytics API error 403: The caller does not have permission"
    )


@pytest.mark.parametrize("status", [429, 403])
def test_used_up_quota_says_to_try_later(status: int):
    error = _http_error(
        status,
        "Exhausted property tokens per day. These quota tokens will return in under "
        "24 hours.",
        "RESOURCE_EXHAUSTED",
    )
    message = str(
        analytics_error(
            error, "block", "id", api=DATA_API, property_name="properties/123456789"
        )
    )
    assert message.startswith(
        "The Google Analytics quota for property 123456789 is used up for now. "
        "Try again later."
    )
    assert "return in under 24 hours" in message


def test_used_up_quota_without_a_property():
    error = _http_error(429, "Quota exceeded.", "RESOURCE_EXHAUSTED")
    message = str(analytics_error(error, "block", "id", api=ADMIN_API))
    assert message.startswith("The Google Analytics API quota is used up for now.")


def test_bad_request_passes_googles_reason_through_with_a_hint():
    reason = (
        "Field pagePathh is not a valid dimension. For a list of valid dimensions "
        "and metrics, see https://developers.google.com/analytics/devguides/"
        "reporting/data/v1/api-schema"
    )
    error = _http_error(400, reason, "INVALID_ARGUMENT")
    message = str(analytics_error(error, "block", "id", api=DATA_API))
    assert message == f"Google Analytics rejected the request: {reason} {FIELDS_HINT}"
    assert "List Dimensions and Metrics" in message

    custom = str(analytics_error(error, "block", "id", api=DATA_API, hint="See X."))
    assert custom.endswith(f"{reason} See X.")
    bare = str(analytics_error(error, "block", "id", api=DATA_API, hint=""))
    assert bare == f"Google Analytics rejected the request: {reason}"


def test_other_errors_keep_the_status_and_reason():
    error = HttpError(httplib2.Response({"status": 503}), b"<html>Unavailable</html>")
    error.resp.reason = "Service Unavailable"
    error.reason = error.resp.reason
    message = str(analytics_error(error, "block", "id", api=DATA_API))
    assert message == "Google Analytics API error 503: Service Unavailable"


def test_errors_keep_the_block_name_and_id():
    error = analytics_error(_http_error(500, "Internal error."), "B", "1", api=DATA_API)
    assert (error.block_name, error.block_id) == ("B", "1")
    assert str(error) == "Google Analytics API error 500: Internal error."


def _http_error(
    status: int, message: str, api_status: str = "", reason: str = ""
) -> HttpError:
    error: dict = {"code": status, "message": message}
    if api_status:
        error["status"] = api_status
    if reason:
        error["details"] = [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": reason,
                "domain": "googleapis.com",
            }
        ]
    content = json.dumps({"error": error}).encode()
    return HttpError(httplib2.Response({"status": status}), content)
