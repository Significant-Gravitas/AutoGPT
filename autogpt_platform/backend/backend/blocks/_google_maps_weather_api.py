"""Weather API models, requests and response parsing for the weather block."""

import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, SecretStr

from backend.blocks._google_maps_api import WEATHER_URL, UnitsSystem, dig, maps_request

HOURS_PER_PAGE = 24


class WeatherMode(str, Enum):
    CURRENT = "current"
    DAILY = "daily"
    HOURLY = "hourly"


class WeatherConditions(BaseModel):
    """Conditions reported for a moment, an hour, or half a day."""

    condition: str | None = Field(
        default=None, description="The weather in words, e.g. 'Partly cloudy'"
    )
    condition_type: str | None = Field(
        default=None,
        description="Google's condition code, e.g. PARTLY_CLOUDY or LIGHT_RAIN",
    )
    humidity_percent: int | None = Field(
        default=None, description="Relative humidity, 0-100"
    )
    uv_index: int | None = Field(default=None, description="UV index")
    precipitation_chance_percent: int | None = Field(
        default=None, description="Chance of precipitation, 0-100"
    )
    precipitation_type: str | None = Field(
        default=None, description="Most likely precipitation, e.g. RAIN or SNOW"
    )
    precipitation_amount: float | None = Field(
        default=None,
        description="Expected precipitation as liquid water, in mm (metric) or inches (imperial)",
    )
    thunderstorm_chance_percent: int | None = Field(
        default=None, description="Chance of thunderstorms, 0-100"
    )
    wind_speed: float | None = Field(
        default=None, description="Wind speed, in km/h (metric) or mph (imperial)"
    )
    wind_gust: float | None = Field(
        default=None, description="Wind gust speed, in km/h or mph"
    )
    wind_direction: str | None = Field(
        default=None, description="Where the wind comes from, e.g. NORTH_NORTHWEST"
    )
    cloud_cover_percent: int | None = Field(
        default=None, description="How much of the sky is cloud, 0-100"
    )


class CurrentWeather(WeatherConditions):
    """The weather at one moment."""

    time: str | None = Field(
        default=None, description="When the weather applies (RFC 3339, UTC)"
    )
    is_daytime: bool | None = Field(default=None, description="Whether it's daytime")
    temperature: float | None = Field(
        default=None, description="Temperature, in °C (metric) or °F (imperial)"
    )
    feels_like: float | None = Field(
        default=None, description="Feels-like temperature, in °C or °F"
    )
    dew_point: float | None = Field(default=None, description="Dew point, in °C or °F")
    visibility: float | None = Field(
        default=None, description="Visibility, in km (metric) or miles (imperial)"
    )
    pressure_millibars: float | None = Field(
        default=None, description="Air pressure at mean sea level, in millibars"
    )


class HourlyForecast(CurrentWeather):
    """The forecast for one hour."""

    local_time: str | None = Field(
        default=None,
        description="Start of the hour in the location's local time, e.g. 2026-09-25T15:00:00+02:00",
    )


class DailyForecast(BaseModel):
    """The forecast for one day."""

    date: str | None = Field(default=None, description="Local date, YYYY-MM-DD")
    max_temperature: float | None = Field(
        default=None, description="High, in °C (metric) or °F (imperial)"
    )
    min_temperature: float | None = Field(default=None, description="Low, in °C or °F")
    feels_like_max: float | None = Field(
        default=None, description="Feels-like high, in °C or °F"
    )
    feels_like_min: float | None = Field(
        default=None, description="Feels-like low, in °C or °F"
    )
    daytime: WeatherConditions | None = Field(
        default=None, description="The day, 7am to 7pm local time"
    )
    nighttime: WeatherConditions | None = Field(
        default=None, description="The night, 7pm to 7am local time"
    )
    sunrise: str | None = Field(default=None, description="Sunrise (RFC 3339, UTC)")
    sunset: str | None = Field(default=None, description="Sunset (RFC 3339, UTC)")
    moon_phase: str | None = Field(
        default=None, description="Moon phase, e.g. WAXING_GIBBOUS"
    )


async def fetch_weather(
    api_key: SecretStr,
    latitude: float,
    longitude: float,
    *,
    mode: WeatherMode,
    days: int,
    hours: int,
    units: UnitsSystem,
    language_code: str = "",
) -> tuple[dict[str, Any], int]:
    """Call the Weather API. Returns the response and how many requests it took."""
    params = {
        "location.latitude": str(latitude),
        "location.longitude": str(longitude),
        "unitsSystem": units.value.upper(),
    }
    if language_code:
        params["languageCode"] = language_code
    if mode == WeatherMode.CURRENT:
        url = f"{WEATHER_URL}/currentConditions:lookup"
        return await maps_request(api_key, "GET", url, params=params), 1
    if mode == WeatherMode.DAILY:
        params.update(days=str(days), pageSize=str(days))
        url = f"{WEATHER_URL}/forecast/days:lookup"
        return await maps_request(api_key, "GET", url, params=params), 1
    return await _fetch_hours(api_key, params, hours)


async def _fetch_hours(
    api_key: SecretStr, params: dict[str, str], hours: int
) -> tuple[dict[str, Any], int]:
    """The hourly forecast arrives 24 hours per page; collect the pages needed."""
    params = {**params, "hours": str(hours), "pageSize": str(HOURS_PER_PAGE)}
    forecast_hours: list[dict[str, Any]] = []
    response: dict[str, Any] = {}
    page_token = ""
    calls = 0
    while calls < math.ceil(hours / HOURS_PER_PAGE):
        page = {**params, "pageToken": page_token} if page_token else params
        response = await maps_request(
            api_key, "GET", f"{WEATHER_URL}/forecast/hours:lookup", params=page
        )
        calls += 1
        forecast_hours.extend(response.get("forecastHours") or [])
        page_token = response.get("nextPageToken", "")
        if not page_token:
            break
    return {
        "forecastHours": forecast_hours[:hours],
        "timeZone": response.get("timeZone"),
    }, calls


def to_current_weather(data: dict[str, Any]) -> CurrentWeather:
    return CurrentWeather(time=data.get("currentTime"), **_moment(data))


def to_hourly_forecast(data: dict[str, Any]) -> HourlyForecast:
    return HourlyForecast(
        time=dig(data, "interval", "startTime"),
        local_time=_local_time(data.get("displayDateTime") or {}),
        **_moment(data),
    )


def to_daily_forecast(data: dict[str, Any]) -> DailyForecast:
    daytime = data.get("daytimeForecast")
    nighttime = data.get("nighttimeForecast")
    return DailyForecast(
        date=_date(data.get("displayDate") or {}),
        max_temperature=dig(data, "maxTemperature", "degrees"),
        min_temperature=dig(data, "minTemperature", "degrees"),
        feels_like_max=dig(data, "feelsLikeMaxTemperature", "degrees"),
        feels_like_min=dig(data, "feelsLikeMinTemperature", "degrees"),
        daytime=WeatherConditions(**_conditions(daytime)) if daytime else None,
        nighttime=WeatherConditions(**_conditions(nighttime)) if nighttime else None,
        sunrise=dig(data, "sunEvents", "sunriseTime"),
        sunset=dig(data, "sunEvents", "sunsetTime"),
        moon_phase=dig(data, "moonEvents", "moonPhase"),
    )


def _conditions(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": dig(data, "weatherCondition", "description", "text"),
        "condition_type": dig(data, "weatherCondition", "type"),
        "humidity_percent": data.get("relativeHumidity"),
        "uv_index": data.get("uvIndex"),
        "precipitation_chance_percent": dig(
            data, "precipitation", "probability", "percent"
        ),
        "precipitation_type": dig(data, "precipitation", "probability", "type"),
        "precipitation_amount": dig(data, "precipitation", "qpf", "quantity"),
        "thunderstorm_chance_percent": data.get("thunderstormProbability"),
        "wind_speed": dig(data, "wind", "speed", "value"),
        "wind_gust": dig(data, "wind", "gust", "value"),
        "wind_direction": dig(data, "wind", "direction", "cardinal"),
        "cloud_cover_percent": data.get("cloudCover"),
    }


def _moment(data: dict[str, Any]) -> dict[str, Any]:
    return {
        **_conditions(data),
        "is_daytime": data.get("isDaytime"),
        "temperature": dig(data, "temperature", "degrees"),
        "feels_like": dig(data, "feelsLikeTemperature", "degrees"),
        "dew_point": dig(data, "dewPoint", "degrees"),
        "visibility": dig(data, "visibility", "distance"),
        "pressure_millibars": dig(data, "airPressure", "meanSeaLevelMillibars"),
    }


def _date(value: dict[str, Any]) -> str | None:
    if not value.get("year"):
        return None
    return f"{value['year']:04d}-{value.get('month', 1):02d}-{value.get('day', 1):02d}"


def _local_time(value: dict[str, Any]) -> str | None:
    """Format a displayDateTime. Google leaves out fields that are zero."""
    date = _date(value)
    if not date:
        return None
    text = f"{date}T{value.get('hours', 0):02d}:{value.get('minutes', 0):02d}:00"
    offset = value.get("utcOffset", "")
    if offset.endswith("s"):
        seconds = int(float(offset[:-1]))
        minutes = abs(seconds) // 60
        sign = "-" if seconds < 0 else "+"
        text += f"{sign}{minutes // 60:02d}:{minutes % 60:02d}"
    return text
