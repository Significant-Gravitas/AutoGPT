"""Unit tests for the Google Maps search block on Places API (New).

The block's own test_input/test_mock case mocks the search away; these cover
the request, paging, parsing, error messages and pricing.
"""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from backend.blocks import google_maps
from backend.blocks.google_maps import (
    SEARCH_FIELDS,
    SEARCH_URL,
    GoogleMapsSearchBlock,
    search_error,
    to_place,
)
from backend.data.model import NodeExecutionStats
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import google_maps_credentials
from backend.util.request import Response

KEY = SecretStr("test-key")


def _response(status: int, body: Any = None) -> Response:
    content = b"" if body is None else body
    if not isinstance(content, bytes):
        content = json.dumps(content).encode()
    raw = MagicMock(status=status, headers={}, reason="Reason")
    return Response(response=raw, url=SEARCH_URL, body=content)


class _FakeRequests:
    """Stands in for the Requests class: records calls, replies in order."""

    def __init__(self, *responses: Response):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, **kwargs: Any) -> "_FakeRequests":
        return self

    async def post(self, url: str, **kwargs: Any) -> Response:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def _page(count: int, token: str = "", start: int = 0) -> Response:
    places = [{"id": f"place-{start + i}"} for i in range(count)]
    body: dict[str, Any] = {"places": places}
    if token:
        body["nextPageToken"] = token
    return _response(200, body)


@pytest.mark.asyncio
async def test_search_request(monkeypatch):
    fake = _FakeRequests(_page(3))
    monkeypatch.setattr(google_maps, "Requests", fake)
    places = await GoogleMapsSearchBlock().search_places(KEY, "pizza in Rome", 5000, 5)
    assert [place.place_id for place in places] == ["place-0", "place-1", "place-2"]
    url, kwargs = fake.calls[0]
    assert url == SEARCH_URL
    assert kwargs["json"] == {"textQuery": "pizza in Rome", "pageSize": 5}
    assert kwargs["headers"] == {
        "X-Goog-Api-Key": "test-key",
        "X-Goog-FieldMask": SEARCH_FIELDS,
    }
    assert kwargs["allow_redirects"] is False
    for field in ("places.nationalPhoneNumber", "places.rating", "nextPageToken"):
        assert field in SEARCH_FIELDS.split(",")


@pytest.mark.asyncio
async def test_search_pages_until_it_has_max_results(monkeypatch):
    fake = _FakeRequests(
        _page(20, "page-2"), _page(20, "page-3", start=20), _page(5, start=40)
    )
    monkeypatch.setattr(google_maps, "Requests", fake)
    places = await GoogleMapsSearchBlock().search_places(KEY, "cafes in Paris", 0, 45)
    assert len(places) == 45
    bodies = [kwargs["json"] for _, kwargs in fake.calls]
    assert bodies == [
        {"textQuery": "cafes in Paris", "pageSize": 20},
        {"textQuery": "cafes in Paris", "pageSize": 20, "pageToken": "page-2"},
        {"textQuery": "cafes in Paris", "pageSize": 5, "pageToken": "page-3"},
    ]


@pytest.mark.asyncio
async def test_search_stops_when_google_has_no_more_pages(monkeypatch):
    fake = _FakeRequests(_page(7))
    monkeypatch.setattr(google_maps, "Requests", fake)
    places = await GoogleMapsSearchBlock().search_places(KEY, "museums", 0, 60)
    assert len(places) == 7
    assert len(fake.calls) == 1


def test_place_fields():
    place = to_place(
        {
            "id": "ChIJ123",
            "displayName": {"text": "Joe's Pizza"},
            "formattedAddress": "7 Carmine St, New York, NY 10014, USA",
            "location": {"latitude": 40.7305, "longitude": -74.0021},
            "googleMapsUri": "https://maps.google.com/?cid=42",
            "nationalPhoneNumber": "(212) 366-1182",
            "websiteUri": "https://www.joespizzanyc.com/",
            "rating": 4.5,
            "userRatingCount": 24310,
        }
    )
    assert (place.name, place.phone, place.reviews) == (
        "Joe's Pizza",
        "(212) 366-1182",
        24310,
    )
    assert (place.place_id, place.latitude, place.longitude) == (
        "ChIJ123",
        40.7305,
        -74.0021,
    )
    assert place.google_maps_url == "https://maps.google.com/?cid=42"


def test_place_fields_google_leaves_out():
    place = to_place({"id": "ChIJ123", "displayName": {"text": "New Place"}})
    assert (place.phone, place.website, place.rating, place.reviews) == ("", "", 0, 0)
    assert place.latitude is None


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (
            403,
            {"error": {"message": "x", "details": [{"reason": "SERVICE_DISABLED"}]}},
            "Places API (New) isn't enabled",
        ),
        (
            400,
            {"error": {"message": "x", "details": [{"reason": "API_KEY_INVALID"}]}},
            "Google rejected the Maps API key",
        ),
        (429, {"error": {"message": "Quota exceeded"}}, "run out of"),
        (
            400,
            {"error": {"message": "Invalid pageSize."}},
            "(HTTP 400): Invalid pageSize.",
        ),
        (502, b"<html>Bad Gateway</html>", "temporary problem (HTTP 502)"),
    ],
)
def test_error_messages(status: int, body: Any, expected: str):
    assert expected in search_error(_response(status, body))


@pytest.mark.asyncio
async def test_errors_stop_the_search(monkeypatch):
    fake = _FakeRequests(
        _response(403, {"error": {"details": [{"reason": "SERVICE_DISABLED"}]}})
    )
    monkeypatch.setattr(google_maps, "Requests", fake)
    with pytest.raises(ValueError, match="isn't enabled"):
        await GoogleMapsSearchBlock().search_places(KEY, "pizza", 0, 5)


@pytest.mark.parametrize(
    "places, credits", [(0, 0), (1, 6), (20, 6), (21, 12), (60, 18)]
)
def test_platform_key_pricing_is_per_page_of_results(places: int, credits: int):
    platform = {
        "credentials": {
            "id": google_maps_credentials.id,
            "provider": google_maps_credentials.provider,
            "type": google_maps_credentials.type,
        }
    }
    block = GoogleMapsSearchBlock()
    stats = NodeExecutionStats(provider_cost=places, provider_cost_type="items")
    assert block_usage_cost(block, platform, stats=stats)[0] == credits
    user_key = {"credentials": google_maps.TEST_CREDENTIALS_INPUT}
    assert block_usage_cost(block, user_key, stats=stats)[0] == 0
