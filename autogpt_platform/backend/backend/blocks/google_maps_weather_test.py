"""Unit tests for the Google Maps weather block and its Weather API helpers.

The block's own test_input/test_mock cases mock the API calls away; these
cover location lookup, request building, paging, parsing and pricing.
"""

from typing import Any

import pytest
from pydantic import SecretStr

from backend.blocks import _google_maps_api, _google_maps_weather_api
from backend.blocks._google_maps_api import (
    PLACE_ESSENTIALS_FIELDS,
    GoogleMapsError,
    MapsLocation,
    UnitsSystem,
    locate,
)
from backend.blocks._google_maps_weather_api import (
    WeatherMode,
    fetch_weather,
    to_current_weather,
    to_daily_forecast,
    to_hourly_forecast,
)
from backend.blocks.google_maps import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google_maps_weather import GoogleMapsWeatherBlock
from backend.data.model import NodeExecutionStats
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import google_maps_credentials
from backend.util.exceptions import BlockExecutionError

KEY = SecretStr("test-key")


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
async def test_locate_reads_coordinates_without_calling_google(monkeypatch):
    fake = _FakeMaps()
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    location, calls = await locate(KEY, " 48.8584 , 2.2945 ")
    assert (location.latitude, location.longitude) == (48.8584, 2.2945)
    assert (calls, fake.calls) == (0, [])


@pytest.mark.asyncio
async def test_locate_searches_for_an_id_then_reads_essentials_details(monkeypatch):
    fake = _FakeMaps(
        {"places": [{"id": "ChIJPlace"}]},
        {
            "id": "ChIJPlace",
            "formattedAddress": "Av. Gustave Eiffel, 75007 Paris, France",
            "location": {"latitude": 48.8583701, "longitude": 2.2944813},
        },
    )
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    location, calls = await locate(KEY, "Eiffel Tower", language_code="fr")

    assert location == MapsLocation(
        latitude=48.8583701,
        longitude=2.2944813,
        address="Av. Gustave Eiffel, 75007 Paris, France",
        place_id="ChIJPlace",
    )
    assert calls == 1
    search, details = fake.calls
    assert search["url"].endswith("/places:searchText")
    assert search["field_mask"] == "places.id"
    assert search["body"] == {
        "textQuery": "Eiffel Tower",
        "pageSize": 1,
        "languageCode": "fr",
    }
    assert details["url"].endswith("/places/ChIJPlace")
    assert details["field_mask"] == PLACE_ESSENTIALS_FIELDS
    assert details["params"] == {"languageCode": "fr"}


@pytest.mark.asyncio
async def test_locate_skips_the_search_for_a_place_id(monkeypatch):
    fake = _FakeMaps({"id": "Ei1abc", "location": {"latitude": 1, "longitude": 2}})
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    location, _ = await locate(KEY, "place_id:Ei1abc")
    assert location.place_id == "Ei1abc"
    assert [call["method"] for call in fake.calls] == ["GET"]


@pytest.mark.asyncio
async def test_locate_explains_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(_google_maps_api, "maps_request", _FakeMaps({}))
    with pytest.raises(GoogleMapsError, match="couldn't find 'Atlantis'"):
        await locate(KEY, "  Atlantis ")


@pytest.mark.asyncio
async def test_current_weather_request(monkeypatch):
    fake = _FakeMaps({"currentTime": "2026-09-25T09:00:00Z"})
    monkeypatch.setattr(_google_maps_weather_api, "maps_request", fake)
    _, calls = await fetch_weather(
        KEY,
        40.7,
        -74.0,
        mode=WeatherMode.CURRENT,
        days=5,
        hours=24,
        units=UnitsSystem.IMPERIAL,
        language_code="es",
    )
    assert calls == 1
    assert fake.calls[0]["url"].endswith("/v1/currentConditions:lookup")
    assert fake.calls[0]["params"] == {
        "location.latitude": "40.7",
        "location.longitude": "-74.0",
        "unitsSystem": "IMPERIAL",
        "languageCode": "es",
    }


@pytest.mark.asyncio
async def test_daily_forecast_asks_for_every_day_in_one_page(monkeypatch):
    fake = _FakeMaps({"forecastDays": []})
    monkeypatch.setattr(_google_maps_weather_api, "maps_request", fake)
    _, calls = await fetch_weather(
        KEY,
        1.0,
        2.0,
        mode=WeatherMode.DAILY,
        days=10,
        hours=24,
        units=UnitsSystem.METRIC,
    )
    assert calls == 1
    assert fake.calls[0]["url"].endswith("/v1/forecast/days:lookup")
    assert fake.calls[0]["params"]["days"] == "10"
    assert fake.calls[0]["params"]["pageSize"] == "10"


@pytest.mark.asyncio
async def test_hourly_forecast_follows_pages_until_it_has_enough_hours(monkeypatch):
    def page(start: int, count: int, token: str = "") -> dict[str, Any]:
        hours = [{"interval": {"startTime": f"hour-{start + i}"}} for i in range(count)]
        response: dict[str, Any] = {
            "forecastHours": hours,
            "timeZone": {"id": "Asia/Kolkata"},
        }
        if token:
            response["nextPageToken"] = token
        return response

    fake = _FakeMaps(page(0, 24, "next-1"), page(24, 24, "next-2"))
    monkeypatch.setattr(_google_maps_weather_api, "maps_request", fake)
    weather, calls = await fetch_weather(
        KEY,
        1.0,
        2.0,
        mode=WeatherMode.HOURLY,
        days=5,
        hours=30,
        units=UnitsSystem.METRIC,
    )
    assert calls == 2
    assert len(weather["forecastHours"]) == 30
    assert weather["forecastHours"][-1]["interval"]["startTime"] == "hour-29"
    assert weather["timeZone"] == {"id": "Asia/Kolkata"}
    assert "pageToken" not in fake.calls[0]["params"]
    assert fake.calls[1]["params"]["pageToken"] == "next-1"
    assert fake.calls[1]["params"]["hours"] == "30"
    assert fake.calls[1]["params"]["pageSize"] == "24"


def test_current_weather_parsing():
    weather = to_current_weather(
        {
            "currentTime": "2026-09-25T09:12:40Z",
            "isDaytime": False,
            "weatherCondition": {
                "description": {"text": "Light rain"},
                "type": "LIGHT_RAIN",
            },
            "temperature": {"degrees": 11.5},
            "feelsLikeTemperature": {"degrees": 9.8},
            "relativeHumidity": 91,
            "precipitation": {
                "probability": {"percent": 80, "type": "RAIN"},
                "qpf": {"quantity": 1.2},
            },
            "wind": {
                "direction": {"cardinal": "SOUTHWEST"},
                "speed": {"value": 22},
                "gust": {"value": 40},
            },
            "visibility": {"distance": 8},
            "airPressure": {"meanSeaLevelMillibars": 1004.2},
        }
    )
    assert weather.condition == "Light rain"
    assert weather.condition_type == "LIGHT_RAIN"
    assert (weather.temperature, weather.feels_like) == (11.5, 9.8)
    assert weather.precipitation_chance_percent == 80
    assert weather.precipitation_amount == 1.2
    assert (weather.wind_speed, weather.wind_gust) == (22, 40)
    assert weather.wind_direction == "SOUTHWEST"
    assert weather.is_daytime is False
    assert weather.uv_index is None


def test_daily_forecast_parsing():
    day = to_daily_forecast(
        {
            "displayDate": {"year": 2026, "month": 1, "day": 2},
            "daytimeForecast": {"weatherCondition": {"type": "SNOW"}},
            "maxTemperature": {"degrees": -1.5},
            "minTemperature": {"degrees": -8},
            "sunEvents": {"sunriseTime": "2026-01-02T07:44:00Z"},
            "moonEvents": {"moonPhase": "FULL_MOON"},
        }
    )
    assert day.date == "2026-01-02"
    assert (day.max_temperature, day.min_temperature) == (-1.5, -8)
    assert day.daytime is not None and day.daytime.condition_type == "SNOW"
    assert day.nighttime is None
    assert day.sunrise == "2026-01-02T07:44:00Z"
    assert day.sunset is None
    assert day.moon_phase == "FULL_MOON"


@pytest.mark.parametrize(
    "display, expected",
    [
        (
            {"year": 2026, "month": 9, "day": 25, "hours": 15, "utcOffset": "7200s"},
            "2026-09-25T15:00:00+02:00",
        ),
        (
            {"year": 2026, "month": 9, "day": 25, "utcOffset": "-12600s"},
            "2026-09-25T00:00:00-03:30",
        ),
        ({"year": 2026, "month": 9, "day": 25, "hours": 6}, "2026-09-25T06:00:00"),
    ],
)
def test_hourly_local_time(display: dict[str, Any], expected: str):
    hour = to_hourly_forecast({"displayDateTime": display})
    assert hour.local_time == expected


@pytest.mark.asyncio
async def test_block_reports_each_billable_call(monkeypatch):
    async def fake_locate(api_key, text, *, language_code=""):
        return MapsLocation(latitude=1, longitude=2), 1

    async def fake_fetch(*args, **kwargs):
        return {"forecastHours": [], "timeZone": {"id": "UTC"}}, 3

    monkeypatch.setattr("backend.blocks.google_maps_weather.locate", fake_locate)
    monkeypatch.setattr("backend.blocks.google_maps_weather.fetch_weather", fake_fetch)
    block = GoogleMapsWeatherBlock()
    input_data = GoogleMapsWeatherBlock.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "location": "Somewhere",
            "mode": "hourly",
            "forecast_hours": 60,
        }
    )
    outputs = [
        name async for name, _ in block.run(input_data, credentials=TEST_CREDENTIALS)
    ]
    assert outputs == ["hours", "location", "time_zone"]
    assert block.execution_stats.provider_cost == 4
    assert block.execution_stats.provider_cost_type == "items"


@pytest.mark.asyncio
async def test_block_turns_api_errors_into_block_errors(monkeypatch):
    async def fake_locate(api_key, text, *, language_code=""):
        raise GoogleMapsError("The Weather API isn't enabled", 403)

    monkeypatch.setattr("backend.blocks.google_maps_weather.locate", fake_locate)
    block = GoogleMapsWeatherBlock()
    input_data = GoogleMapsWeatherBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "location": "Paris"}
    )
    with pytest.raises(BlockExecutionError, match="isn't enabled"):
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"mode": "daily", "forecast_days": 10},
        {"mode": "hourly", "forecast_hours": 240},
    ],
)
def test_platform_key_costs_one_credit_in_every_mode(extra: dict[str, Any]):
    platform = {
        "credentials": {
            "id": google_maps_credentials.id,
            "provider": google_maps_credentials.provider,
            "type": google_maps_credentials.type,
        }
    }
    block = GoogleMapsWeatherBlock()
    assert block_usage_cost(block, {**platform, **extra})[0] == 1
    assert (
        block_usage_cost(block, {"credentials": TEST_CREDENTIALS_INPUT, **extra})[0]
        == 0
    )
    stats = NodeExecutionStats(provider_cost=11, provider_cost_type="items")
    assert block_usage_cost(block, {**platform, **extra}, stats=stats)[0] == 1
