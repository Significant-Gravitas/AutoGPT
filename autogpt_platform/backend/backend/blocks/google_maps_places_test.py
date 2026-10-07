"""Unit tests for the Google Maps resolve-places and resolve-links blocks.

The blocks' own test_input/test_mock cases mock the API calls away; these
cover the Places requests, per-item failures, lookup counts and pricing.
"""

from typing import Any

import pytest
from pydantic import SecretStr

from backend.blocks import _google_maps_api
from backend.blocks._google_maps_api import PLACE_PRO_FIELDS, GoogleMapsError
from backend.blocks.google_maps import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google_maps_places import (
    GoogleMapsResolveLinksBlock,
    GoogleMapsResolvePlacesBlock,
    resolve_query,
)
from backend.data.model import NodeExecutionStats
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import google_maps_credentials
from backend.util.exceptions import BlockExecutionError

KEY = SecretStr("test-key")
PLACE = {
    "id": "ChIJLU7jZClu5kcR4PcOOO6p3I0",
    "displayName": {"text": "Eiffel Tower"},
    "formattedAddress": "Av. Gustave Eiffel, 75007 Paris, France",
    "location": {"latitude": 48.8583701, "longitude": 2.2944813},
    "types": ["tourist_attraction"],
    "googleMapsUri": "https://maps.google.com/?cid=10222232094831998944",
}
PLACE_URL = (
    "https://www.google.com/maps/place/Eiffel+Tower/@48.85837,2.29448,17z/"
    "data=!4m6!3m5!1s0x47e66e2964e34e2d:0x8ddca9ee380ef7e0!8m2"
    "!3d48.8583701!4d2.2944813!16zL20vMDJqODE"
)


class _FakeMaps:
    """Stands in for maps_request: records calls, replies in order."""

    def __init__(self, *responses: dict[str, Any] | Exception):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, api_key, method, url, *, params=None, body=None, field_mask=""
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "body": body,
                "field_mask": field_mask,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
async def test_resolve_query_searches_ids_then_reads_pro_details(monkeypatch):
    fake = _FakeMaps({"places": [{"id": "ChIJPlace"}]}, PLACE)
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    assert await resolve_query(KEY, "Eiffel Tower", "FR") == PLACE
    search, details = fake.calls
    assert search["field_mask"] == "places.id"
    assert search["body"] == {
        "textQuery": "Eiffel Tower",
        "pageSize": 1,
        "regionCode": "FR",
    }
    assert details["url"].endswith("/places/ChIJPlace")
    assert details["field_mask"] == PLACE_PRO_FIELDS
    assert details["params"] == {"regionCode": "FR"}


@pytest.mark.asyncio
async def test_resolve_query_takes_a_place_id_as_is(monkeypatch):
    fake = _FakeMaps(PLACE)
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    await resolve_query(KEY, "ChIJLU7jZClu5kcR4PcOOO6p3I0")
    assert [call["method"] for call in fake.calls] == ["GET"]


@pytest.mark.asyncio
async def test_resolve_query_without_a_match(monkeypatch):
    monkeypatch.setattr(_google_maps_api, "maps_request", _FakeMaps({}))
    assert await resolve_query(KEY, "Nowhere Special 12345") is None
    assert await resolve_query(KEY, "") is None


@pytest.mark.asyncio
async def test_resolve_query_treats_an_unknown_place_id_as_no_match(monkeypatch):
    fake = _FakeMaps(GoogleMapsError("Not found", 404))
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    assert await resolve_query(KEY, "place_id:gone") is None


@pytest.mark.asyncio
async def test_resolve_places_block_fails_on_api_errors(monkeypatch):
    fake = _FakeMaps(GoogleMapsError("The Places API (New) isn't enabled", 403))
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    block = GoogleMapsResolvePlacesBlock()
    input_data = GoogleMapsResolvePlacesBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["Eiffel Tower"]}
    )
    with pytest.raises(BlockExecutionError, match="isn't enabled"):
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass


@pytest.mark.asyncio
async def test_resolve_places_block_counts_each_lookup(monkeypatch):
    async def fake_resolve(api_key, query, region_code=""):
        return PLACE if query else None

    monkeypatch.setattr("backend.blocks.google_maps_places.resolve_query", fake_resolve)
    block = GoogleMapsResolvePlacesBlock()
    input_data = GoogleMapsResolvePlacesBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["A", " ", "B"]}
    )
    outputs = [o async for o in block.run(input_data, credentials=TEST_CREDENTIALS)]
    places = dict(outputs)["places"]
    assert [place.query for place in places] == ["A", "B"]
    assert dict(outputs)["unresolved"] == [""]
    assert block.execution_stats.provider_cost == 2


def test_input_takes_one_to_twenty_queries():
    make = GoogleMapsResolvePlacesBlock.Input.model_validate
    with pytest.raises(ValueError):
        make({"credentials": TEST_CREDENTIALS_INPUT, "queries": []})
    with pytest.raises(ValueError):
        make({"credentials": TEST_CREDENTIALS_INPUT, "queries": ["x"] * 21})


async def _resolve_links(block: GoogleMapsResolveLinksBlock, **fields: Any) -> dict:
    input_data = GoogleMapsResolveLinksBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **fields}
    )
    outputs = [o async for o in block.run(input_data, credentials=TEST_CREDENTIALS)]
    return {name: value for name, value in outputs if name != "result"}


@pytest.mark.asyncio
async def test_links_can_be_read_without_a_lookup(monkeypatch):
    fake = _FakeMaps()
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    block = GoogleMapsResolveLinksBlock()
    outputs = await _resolve_links(block, urls=[PLACE_URL], look_up_place=False)
    result = outputs["results"][0]
    assert (result.name, result.latitude, result.place_id) == (
        "Eiffel Tower",
        48.8583701,
        "",
    )
    assert fake.calls == []
    assert block.execution_stats.provider_cost == 0


@pytest.mark.asyncio
async def test_map_view_links_give_coordinates_without_a_lookup(monkeypatch):
    fake = _FakeMaps()
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    outputs = await _resolve_links(
        GoogleMapsResolveLinksBlock(),
        urls=["https://www.google.com/maps/@51.5007,-0.1246,15z"],
    )
    result = outputs["results"][0]
    assert (result.latitude, result.longitude, result.place_id) == (
        51.5007,
        -0.1246,
        "",
    )
    assert fake.calls == []


@pytest.mark.asyncio
async def test_lookups_are_counted_and_misses_explained(monkeypatch):
    fake = _FakeMaps({"places": [{"id": "ChIJPlace"}]}, PLACE, {})
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    block = GoogleMapsResolveLinksBlock()
    missing = "https://www.google.com/maps/place/Nowhere/@1.5,2.5,17z/data=!3d1.5!4d2.5"
    outputs = await _resolve_links(block, urls=[PLACE_URL, missing, "example.com"])
    assert [result.place_id for result in outputs["results"]] == [PLACE["id"]]
    assert [failure.url for failure in outputs["failed"]] == [missing, "example.com"]
    assert (
        "no place called 'Nowhere' at the link's location"
        in outputs["failed"][0].reason
    )
    assert "isn't a Google Maps link" in outputs["failed"][1].reason
    assert block.execution_stats.provider_cost == 1


@pytest.mark.asyncio
async def test_resolve_links_block_fails_on_api_errors(monkeypatch):
    fake = _FakeMaps(GoogleMapsError("This Maps API key has run out of quota", 429))
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    with pytest.raises(BlockExecutionError, match="run out of quota"):
        await _resolve_links(GoogleMapsResolveLinksBlock(), urls=[PLACE_URL])


@pytest.mark.parametrize(
    "block_class", [GoogleMapsResolvePlacesBlock, GoogleMapsResolveLinksBlock]
)
def test_platform_key_charges_three_credits_per_place_looked_up(block_class):
    platform = {
        "credentials": {
            "id": google_maps_credentials.id,
            "provider": google_maps_credentials.provider,
            "type": google_maps_credentials.type,
        }
    }
    user_key = {"credentials": TEST_CREDENTIALS_INPUT}
    block = block_class()
    stats = NodeExecutionStats(provider_cost=4, provider_cost_type="items")
    assert block_usage_cost(block, platform, stats=stats)[0] == 12
    assert block_usage_cost(block, user_key, stats=stats)[0] == 0
    no_lookups = NodeExecutionStats(provider_cost=0, provider_cost_type="items")
    assert block_usage_cost(block, platform, stats=no_lookups)[0] == 0
