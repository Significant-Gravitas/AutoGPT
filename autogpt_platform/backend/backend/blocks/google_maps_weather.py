from typing import Any, Iterator

from pydantic import SecretStr

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks._google_maps_api import (
    GoogleMapsCredentialsField,
    GoogleMapsCredentialsInput,
    GoogleMapsError,
    MapsLocation,
    UnitsSystem,
    dig,
    locate,
)
from backend.blocks._google_maps_weather_api import (
    CurrentWeather,
    DailyForecast,
    HourlyForecast,
    WeatherMode,
    fetch_weather,
    to_current_weather,
    to_daily_forecast,
    to_hourly_forecast,
)
from backend.blocks.google_maps import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.data.model import APIKeyCredentials, NodeExecutionStats, SchemaField
from backend.util.exceptions import BlockExecutionError, BlockInputError

_TEST_LOCATION = MapsLocation(
    latitude=48.8583701,
    longitude=2.2944813,
    address="Av. Gustave Eiffel, 75007 Paris, France",
    place_id="ChIJLU7jZClu5kcR4PcOOO6p3I0",
)
_TEST_CONDITIONS = {
    "weatherCondition": {
        "description": {"text": "Partly cloudy", "languageCode": "en"},
        "type": "PARTLY_CLOUDY",
    },
    "relativeHumidity": 64,
    "uvIndex": 3,
    "precipitation": {
        "probability": {"percent": 10, "type": "RAIN"},
        "qpf": {"quantity": 0, "unit": "MILLIMETERS"},
    },
    "thunderstormProbability": 0,
    "wind": {
        "direction": {"degrees": 250, "cardinal": "WEST_SOUTHWEST"},
        "speed": {"value": 14, "unit": "KILOMETERS_PER_HOUR"},
        "gust": {"value": 27, "unit": "KILOMETERS_PER_HOUR"},
    },
    "cloudCover": 45,
}
_TEST_MOMENT = {
    **_TEST_CONDITIONS,
    "isDaytime": True,
    "temperature": {"degrees": 18.2, "unit": "CELSIUS"},
    "feelsLikeTemperature": {"degrees": 17.6, "unit": "CELSIUS"},
    "dewPoint": {"degrees": 11.3, "unit": "CELSIUS"},
    "visibility": {"distance": 16, "unit": "KILOMETERS"},
    "airPressure": {"meanSeaLevelMillibars": 1016.4},
}
_TEST_DAY = {
    "displayDate": {"year": 2026, "month": 9, "day": 25},
    "daytimeForecast": _TEST_CONDITIONS,
    "nighttimeForecast": {**_TEST_CONDITIONS, "uvIndex": 0},
    "maxTemperature": {"degrees": 21.4, "unit": "CELSIUS"},
    "minTemperature": {"degrees": 12.9, "unit": "CELSIUS"},
    "sunEvents": {
        "sunriseTime": "2026-09-25T05:38:12Z",
        "sunsetTime": "2026-09-25T17:34:40Z",
    },
    "moonEvents": {"moonPhase": "WAXING_GIBBOUS"},
}
_TEST_HOUR = {
    **_TEST_MOMENT,
    "interval": {"startTime": "2026-09-25T09:00:00Z"},
    "displayDateTime": {
        "year": 2026,
        "month": 9,
        "day": 25,
        "hours": 11,
        "utcOffset": "7200s",
    },
}
# One canned response carries every mode's fields; each mode reads its own.
_TEST_WEATHER = {
    **_TEST_MOMENT,
    "currentTime": "2026-09-25T09:12:40Z",
    "timeZone": {"id": "Europe/Paris"},
    "forecastDays": [_TEST_DAY],
    "forecastHours": [_TEST_HOUR],
}


class GoogleMapsWeatherBlock(Block):
    """Current weather, or a daily or hourly forecast, for any place."""

    class Input(BlockSchemaInput):
        credentials: GoogleMapsCredentialsInput = GoogleMapsCredentialsField()
        location: str = SchemaField(
            description=(
                "Where to get the weather for: an address, a place name, "
                "'latitude,longitude' or a Google Maps place ID"
            ),
            placeholder="e.g. 'Eiffel Tower, Paris' or '48.8584,2.2945'",
        )
        mode: WeatherMode = SchemaField(
            description=(
                "current: the weather right now. daily: a forecast for each day. "
                "hourly: a forecast for each hour."
            ),
            default=WeatherMode.CURRENT,
        )
        forecast_days: int = SchemaField(
            description="Days to forecast, starting today (daily mode)",
            default=5,
            ge=1,
            le=10,
        )
        forecast_hours: int = SchemaField(
            description="Hours to forecast, starting with the current hour (hourly mode)",
            default=24,
            ge=1,
            le=240,
        )
        units: UnitsSystem = SchemaField(
            description="metric (°C, km/h, mm, km) or imperial (°F, mph, inches, miles)",
            default=UnitsSystem.METRIC,
        )
        language_code: str = SchemaField(
            description="Language for weather descriptions and the address, e.g. 'en' or 'fr'. Defaults to English.",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        current: CurrentWeather = SchemaField(
            description="The weather right now (current mode)"
        )
        days: list[DailyForecast] = SchemaField(
            description="One forecast per day, starting today (daily mode)"
        )
        day: DailyForecast = SchemaField(description="Each day's forecast (daily mode)")
        hours: list[HourlyForecast] = SchemaField(
            description="One forecast per hour, starting with the current hour (hourly mode)"
        )
        hour: HourlyForecast = SchemaField(
            description="Each hour's forecast (hourly mode)"
        )
        location: MapsLocation = SchemaField(
            description="The place the weather is for, with its coordinates"
        )
        time_zone: str = SchemaField(
            description="The location's time zone, e.g. Europe/Paris"
        )

    def __init__(self):
        test_day = to_daily_forecast(_TEST_DAY)
        test_hour = to_hourly_forecast(_TEST_HOUR)
        where = [("location", _TEST_LOCATION), ("time_zone", "Europe/Paris")]
        super().__init__(
            id="7161cfec-6a75-40d4-ab5d-4eb428c6932f",
            description=(
                "Get the weather for a place from Google Maps: current conditions, "
                "a daily forecast for up to 10 days, or an hourly forecast for up "
                "to 240 hours. The place can be an address, a place name, "
                "coordinates or a place ID."
            ),
            categories={BlockCategory.SEARCH},
            input_schema=GoogleMapsWeatherBlock.Input,
            output_schema=GoogleMapsWeatherBlock.Output,
            test_input=[
                {"credentials": TEST_CREDENTIALS_INPUT, "location": "Eiffel Tower"},
                {
                    "credentials": TEST_CREDENTIALS_INPUT,
                    "location": "Eiffel Tower",
                    "mode": WeatherMode.DAILY,
                    "forecast_days": 1,
                },
                {
                    "credentials": TEST_CREDENTIALS_INPUT,
                    "location": "48.8583701,2.2944813",
                    "mode": WeatherMode.HOURLY,
                    "forecast_hours": 1,
                },
            ],
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("current", to_current_weather(_TEST_WEATHER)),
                *where,
                ("days", [test_day]),
                ("day", test_day),
                *where,
                ("hours", [test_hour]),
                ("hour", test_hour),
                *where,
            ],
            test_mock={
                "_locate": lambda *args, **kwargs: (_TEST_LOCATION, 1),
                "_fetch_weather": lambda *args, **kwargs: (_TEST_WEATHER, 1),
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.location.strip():
            raise BlockInputError(
                message="Give a location: an address, a place name, coordinates or a place ID.",
                block_name=self.name,
                block_id=self.id,
            )
        try:
            location, lookups = await self._locate(
                credentials.api_key, input_data.location, input_data.language_code
            )
            weather, calls = await self._fetch_weather(
                credentials.api_key, location, input_data
            )
        except GoogleMapsError as e:
            raise BlockExecutionError(
                message=str(e), block_name=self.name, block_id=self.id
            ) from e
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=float(lookups + calls), provider_cost_type="items"
            )
        )
        for name, value in weather_outputs(input_data.mode, weather):
            yield name, value
        yield "location", location
        if time_zone := dig(weather, "timeZone", "id"):
            yield "time_zone", time_zone

    @staticmethod
    async def _locate(
        api_key: SecretStr, text: str, language_code: str
    ) -> tuple[MapsLocation, int]:
        return await locate(api_key, text, language_code=language_code)

    @staticmethod
    async def _fetch_weather(
        api_key: SecretStr,
        location: MapsLocation,
        input_data: "GoogleMapsWeatherBlock.Input",
    ) -> tuple[dict[str, Any], int]:
        return await fetch_weather(
            api_key,
            location.latitude,
            location.longitude,
            mode=input_data.mode,
            days=input_data.forecast_days,
            hours=input_data.forecast_hours,
            units=input_data.units,
            language_code=input_data.language_code,
        )


def weather_outputs(
    mode: WeatherMode, weather: dict[str, Any]
) -> Iterator[tuple[str, Any]]:
    """The mode's outputs: `current`, or a list plus one output per day or hour."""
    if mode == WeatherMode.CURRENT:
        yield "current", to_current_weather(weather)
    elif mode == WeatherMode.DAILY:
        days = [to_daily_forecast(day) for day in weather.get("forecastDays") or []]
        yield "days", days
        yield from (("day", day) for day in days)
    else:
        hours = [to_hourly_forecast(h) for h in weather.get("forecastHours") or []]
        yield "hours", hours
        yield from (("hour", hour) for hour in hours)
