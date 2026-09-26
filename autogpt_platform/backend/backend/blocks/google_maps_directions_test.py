"""Unit tests for the Google Maps directions block and its Routes API helpers.

The block's own test_input/test_mock cases mock the API call away; these
cover request building, field masks, parsing, input checks and pricing.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import SecretStr

from backend.blocks import _google_maps_routes_api
from backend.blocks._google_maps_api import GoogleMapsError, UnitsSystem
from backend.blocks._google_maps_routes_api import (
    TravelMode,
    build_route_request,
    compute_route,
    route_outputs,
    route_summary,
    to_waypoint,
)
from backend.blocks.google_maps import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google_maps_directions import GoogleMapsGetDirectionsBlock
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import google_maps_credentials
from backend.util.exceptions import BlockExecutionError, BlockInputError

PLATFORM_KEY = {
    "credentials": {
        "id": google_maps_credentials.id,
        "provider": google_maps_credentials.provider,
        "type": google_maps_credentials.type,
    }
}


def _request(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "origin": "Eiffel Tower, Paris",
        "destination": "Louvre Museum, Paris",
        "travel_mode": TravelMode.DRIVE,
        "use_live_traffic": False,
        "units": UnitsSystem.METRIC,
    }
    return build_route_request(**{**args, **overrides})


def _input(**fields: Any) -> GoogleMapsGetDirectionsBlock.Input:
    return GoogleMapsGetDirectionsBlock.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "origin": "Eiffel Tower, Paris",
            "destination": "Louvre Museum, Paris",
            **fields,
        }
    )


async def _run(block: GoogleMapsGetDirectionsBlock, **fields: Any) -> list:
    return [
        output
        async for output in block.run(_input(**fields), credentials=TEST_CREDENTIALS)
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Eiffel Tower, Paris ", {"address": "Eiffel Tower, Paris"}),
        (
            "48.8584,2.2945",
            {"location": {"latLng": {"latitude": 48.8584, "longitude": 2.2945}}},
        ),
        ("ChIJLU7jZClu5kcR4PcOOO6p3I0", {"placeId": "ChIJLU7jZClu5kcR4PcOOO6p3I0"}),
        ("place_id:EiRSdWUgZGU", {"placeId": "EiRSdWUgZGU"}),
        ("95,200", {"address": "95,200"}),
    ],
)
def test_waypoints(text: str, expected: dict[str, Any]):
    assert to_waypoint(text) == expected


def test_drive_routes_ignore_traffic_unless_asked():
    body = _request()
    assert body["travelMode"] == "DRIVE"
    assert body["routingPreference"] == "TRAFFIC_UNAWARE"
    assert body["units"] == "METRIC"
    assert "departureTime" not in body


def test_live_traffic_uses_the_traffic_aware_preference():
    assert _request(use_live_traffic=True)["routingPreference"] == "TRAFFIC_AWARE"
    two_wheeler = _request(travel_mode=TravelMode.TWO_WHEELER)
    assert two_wheeler["travelMode"] == "TWO_WHEELER"
    assert two_wheeler["routingPreference"] == "TRAFFIC_UNAWARE"


@pytest.mark.parametrize(
    "mode", [TravelMode.WALK, TravelMode.BICYCLE, TravelMode.TRANSIT]
)
def test_other_modes_send_no_routing_preference(mode: TravelMode):
    assert "routingPreference" not in _request(travel_mode=mode)


def test_departure_time_is_sent_for_transit_in_utc():
    leave = datetime(2026, 9, 26, 8, 30, tzinfo=timezone(timedelta(hours=2)))
    body = _request(travel_mode=TravelMode.TRANSIT, departure_time=leave)
    assert body["departureTime"] == "2026-09-26T06:30:00Z"


def test_departure_time_without_a_zone_is_read_as_utc():
    body = _request(use_live_traffic=True, departure_time=datetime(2026, 9, 26, 8))
    assert body["departureTime"] == "2026-09-26T08:00:00Z"


def test_departure_time_is_left_out_when_it_would_do_nothing():
    assert "departureTime" not in _request(departure_time=datetime(2026, 9, 26, 8))


@pytest.mark.asyncio
@pytest.mark.parametrize("include_steps", [False, True])
async def test_field_mask_only_asks_for_steps_when_needed(monkeypatch, include_steps):
    calls: list[dict[str, Any]] = []

    async def fake_request(api_key, method, url, *, body=None, field_mask="", **_):
        calls.append({"method": method, "url": url, "mask": field_mask})
        return {}

    monkeypatch.setattr(_google_maps_routes_api, "maps_request", fake_request)
    await compute_route(SecretStr("k"), {}, include_steps=include_steps)
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"].endswith("/directions/v2:computeRoutes")
    assert "routes.duration" in calls[0]["mask"]
    assert ("routes.legs.steps.navigationInstruction" in calls[0]["mask"]) is (
        include_steps
    )


def test_transit_steps_are_parsed():
    route = {
        "distanceMeters": 9100,
        "duration": "1500.4s",
        "localizedValues": {
            "distance": {"text": "9.1 km"},
            "duration": {"text": "25 mins"},
        },
        "warnings": ["Check the timetable"],
        "legs": [
            {
                "steps": [
                    {
                        "travelMode": "TRANSIT",
                        "distanceMeters": 8000,
                        "staticDuration": "960s",
                        "transitDetails": {
                            "headsign": "La Defense",
                            "stopCount": 6,
                            "stopDetails": {
                                "departureStop": {"name": "Concorde"},
                                "arrivalStop": {"name": "Charles de Gaulle - Etoile"},
                            },
                            "localizedValues": {
                                "departureTime": {"time": {"text": "8:05 AM"}},
                                "arrivalTime": {"time": {"text": "8:21 AM"}},
                            },
                            "transitLine": {
                                "name": "Metro Line 1",
                                "nameShort": "1",
                                "vehicle": {"name": {"text": "Subway"}},
                            },
                        },
                    }
                ]
            }
        ],
    }
    outputs = dict(route_outputs(route, include_steps=True))
    assert outputs["summary"] == "25 mins (9.1 km)"
    assert outputs["duration_seconds"] == 1500
    assert outputs["warnings"] == ["Check the timetable"]
    transit = outputs["step"].transit
    assert transit is not None
    assert (transit.line, transit.vehicle, transit.headsign) == (
        "1",
        "Subway",
        "La Defense",
    )
    assert (transit.departure_stop, transit.arrival_stop) == (
        "Concorde",
        "Charles de Gaulle - Etoile",
    )
    assert (transit.departure_time, transit.arrival_time, transit.stop_count) == (
        "8:05 AM",
        "8:21 AM",
        6,
    )


def test_summary_works_without_localized_text():
    route = {"distanceMeters": 12380, "duration": "1260s", "description": "I-280 S"}
    assert route_summary(route) == "21 min (12.4 km) via I-280 S"


@pytest.mark.asyncio
async def test_live_traffic_is_refused_for_walking():
    with pytest.raises(BlockInputError, match="drive and two-wheeler"):
        await _run(
            GoogleMapsGetDirectionsBlock(), travel_mode="walk", use_live_traffic=True
        )


@pytest.mark.asyncio
async def test_blank_destination_is_refused():
    with pytest.raises(BlockInputError, match="origin and a destination"):
        await _run(GoogleMapsGetDirectionsBlock(), destination="  ")


@pytest.mark.asyncio
async def test_no_route_is_an_error(monkeypatch):
    block = GoogleMapsGetDirectionsBlock()

    async def no_routes(*args, **kwargs):
        return {}

    monkeypatch.setattr(block, "_compute_route", no_routes)
    with pytest.raises(BlockExecutionError, match="no cycling route"):
        await _run(block, travel_mode="bicycle")
    assert block.execution_stats.provider_cost == 1


@pytest.mark.asyncio
async def test_api_errors_become_block_errors(monkeypatch):
    block = GoogleMapsGetDirectionsBlock()

    async def disabled(*args, **kwargs):
        raise GoogleMapsError("The Routes API isn't enabled", 403)

    monkeypatch.setattr(block, "_compute_route", disabled)
    with pytest.raises(BlockExecutionError, match="Routes API isn't enabled"):
        await _run(block)


@pytest.mark.parametrize(
    "fields, credits",
    [
        ({}, 1),
        ({"travel_mode": "walk"}, 1),
        ({"use_live_traffic": True}, 2),
        ({"travel_mode": "two_wheeler"}, 3),
        ({"travel_mode": "two_wheeler", "use_live_traffic": True}, 3),
    ],
)
def test_platform_key_pricing(fields: dict[str, Any], credits: int):
    block = GoogleMapsGetDirectionsBlock()
    assert block_usage_cost(block, {**PLATFORM_KEY, **fields})[0] == credits
    user_key = {"credentials": TEST_CREDENTIALS_INPUT, **fields}
    assert block_usage_cost(block, user_key)[0] == 0
