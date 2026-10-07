"""Unit tests for the shared Google Maps Platform helpers: the HTTP call, its
error messages, and reading coordinates and place IDs from text."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from backend.blocks import _google_maps_api
from backend.blocks._google_maps_api import (
    GoogleMapsError,
    maps_error,
    maps_request,
    parse_lat_lng,
    parse_place_id,
)
from backend.util.request import Response

KEY = SecretStr("test-key")
SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"


def _response(status: int, body: Any = None) -> Response:
    content = b"" if body is None else body
    if not isinstance(content, bytes):
        content = json.dumps(content).encode()
    raw = MagicMock(status=status, headers={}, reason="Reason")
    return Response(response=raw, url="https://example.invalid", body=content)


class _FakeRequests:
    """Stands in for the Requests class: records calls, replies in order."""

    def __init__(self, *responses: Response):
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def __call__(self, **kwargs: Any) -> "_FakeRequests":
        return self

    async def request(self, method: str, url: str, **kwargs: Any) -> Response:
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def _error(status: int, reason: str = "", message: str = "Something failed") -> dict:
    details = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}]
    return {"error": {"code": status, "message": message, "details": details}}


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (400, _error(400, "API_KEY_INVALID"), "Google rejected the Maps API key"),
        (401, None, "Google rejected the Maps API key"),
        (
            403,
            _error(403, "SERVICE_DISABLED"),
            "The Places API (New) isn't enabled on the Google Cloud project",
        ),
        (403, _error(403, "BILLING_DISABLED"), "needs billing"),
        (
            403,
            _error(403, "API_KEY_SERVICE_BLOCKED"),
            "isn't allowed to call the Places API (New)",
        ),
        (403, _error(403, "API_KEY_HTTP_REFERRER_BLOCKED"), "certain websites"),
        (429, _error(429, "RATE_LIMIT_EXCEEDED"), "run out of Places API (New) quota"),
        (503, b"<html>Unavailable</html>", "temporary problem (HTTP 503)"),
        (
            400,
            _error(400, message="Invalid textQuery."),
            "(HTTP 400): Invalid textQuery.",
        ),
    ],
)
def test_error_messages(status: int, body: Any, expected: str):
    error = maps_error(SEARCH_URL, _response(status, body))
    assert expected in str(error)
    assert error.status == status


@pytest.mark.asyncio
async def test_requests_put_the_key_in_a_header(monkeypatch):
    fake = _FakeRequests(_response(200, {"places": []}))
    monkeypatch.setattr(_google_maps_api, "Requests", fake)
    result = await maps_request(
        KEY, "POST", SEARCH_URL, body={"textQuery": "x"}, field_mask="places.id"
    )
    assert result == {"places": []}
    method, url, kwargs = fake.calls[0]
    assert (method, url) == ("POST", SEARCH_URL)
    assert kwargs["headers"] == {
        "X-Goog-Api-Key": "test-key",
        "X-Goog-FieldMask": "places.id",
    }
    assert kwargs["json"] == {"textQuery": "x"}
    assert kwargs["allow_redirects"] is False


@pytest.mark.asyncio
async def test_request_errors_raise_google_maps_errors(monkeypatch):
    fake = _FakeRequests(_response(403, _error(403, "SERVICE_DISABLED")))
    monkeypatch.setattr(_google_maps_api, "Requests", fake)
    with pytest.raises(GoogleMapsError, match="Weather API isn't enabled"):
        await maps_request(KEY, "GET", "https://weather.googleapis.com/v1/x")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("48.8584,2.2945", (48.8584, 2.2945)),
        (" -33.86 , +151.21 ", (-33.86, 151.21)),
        ("91,0", None),
        ("48.8584", None),
        ("Paris, France", None),
    ],
)
def test_parse_lat_lng(text: str, expected: tuple[float, float] | None):
    assert parse_lat_lng(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ChIJLU7jZClu5kcR4PcOOO6p3I0", "ChIJLU7jZClu5kcR4PcOOO6p3I0"),
        ("place_id: EiRSdWUgZGU", "EiRSdWUgZGU"),
        ("Eiffel Tower", None),
        ("ChIJshort", None),
    ],
)
def test_parse_place_id(text: str, expected: str | None):
    assert parse_place_id(text) == expected
