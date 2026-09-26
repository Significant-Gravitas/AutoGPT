"""Shared helpers for the Google Maps Platform blocks.

The blocks call Google's REST APIs (Places API (New), Routes API, Weather API)
with the `google_maps` API key. The key goes in the X-Goog-Api-Key header, so
it never appears in a URL.
"""

import re
from enum import Enum
from typing import Any, Literal
from urllib.parse import quote, urlparse

from pydantic import BaseModel, Field, SecretStr

from backend.data.model import CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName
from backend.util.request import Requests, Response

PLACES_URL = "https://places.googleapis.com/v1"
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
WEATHER_URL = "https://weather.googleapis.com/v1"

# Place Details field masks. The fields asked for set the price: Essentials
# has the address and coordinates, Pro adds the name and the Maps link.
PLACE_ESSENTIALS_FIELDS = "id,formattedAddress,location"
PLACE_PRO_FIELDS = "id,displayName,formattedAddress,location,types,googleMapsUri"

_API_NAMES = {
    "places.googleapis.com": "Places API (New)",
    "routes.googleapis.com": "Routes API",
    "weather.googleapis.com": "Weather API",
}
_REASON_MESSAGES = [
    (
        {"API_KEY_INVALID"},
        "Google rejected the Maps API key. Check the key saved in your Google "
        "Maps credentials.",
    ),
    (
        {"SERVICE_DISABLED", "ACCESS_NOT_CONFIGURED"},
        "The {api} isn't enabled on the Google Cloud project that owns this Maps "
        "API key. Enable it in the Google Cloud console under APIs & Services, "
        "then try again.",
    ),
    (
        {"BILLING_DISABLED"},
        "The {api} needs billing, and billing isn't enabled on the Google Cloud "
        "project that owns this Maps API key.",
    ),
    (
        {"API_KEY_SERVICE_BLOCKED"},
        "This Maps API key isn't allowed to call the {api}. Add the {api} to the "
        "key's API restrictions in the Google Cloud console.",
    ),
    (
        {
            "API_KEY_HTTP_REFERRER_BLOCKED",
            "API_KEY_IP_ADDRESS_BLOCKED",
            "API_KEY_ANDROID_APP_BLOCKED",
            "API_KEY_IOS_APP_BLOCKED",
        },
        "This Maps API key only works from certain websites, apps or IP "
        "addresses, so Google blocked the request. Use a key without website "
        "or app restrictions.",
    ),
]

_LAT_LNG = re.compile(r"\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*")
_PLACE_ID = re.compile(r"(?:ChIJ|GhIJ)[A-Za-z0-9_-]{20,}")

GoogleMapsCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.GOOGLE_MAPS], Literal["api_key"]
]


def GoogleMapsCredentialsField() -> GoogleMapsCredentialsInput:
    return CredentialsField(description="Google Maps API key")


class UnitsSystem(str, Enum):
    METRIC = "metric"
    IMPERIAL = "imperial"


class GoogleMapsError(Exception):
    """A Maps Platform API error, with a message the user can act on."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class MapsLocation(BaseModel):
    """A point on the map, and the place it was looked up from."""

    latitude: float = Field(description="Latitude in degrees")
    longitude: float = Field(description="Longitude in degrees")
    address: str = Field(
        default="",
        description="Full address, when the location was given as an address, name or place ID",
    )
    place_id: str = Field(default="", description="Google Maps place ID, when known")


class MapsPlace(BaseModel):
    """A place on Google Maps."""

    place_id: str = Field(default="", description="Google Maps place ID")
    name: str = Field(default="", description="Name of the place")
    address: str = Field(default="", description="Full address")
    latitude: float | None = Field(default=None, description="Latitude in degrees")
    longitude: float | None = Field(default=None, description="Longitude in degrees")
    types: list[str] = Field(
        default_factory=list,
        description="Place types, e.g. restaurant or tourist_attraction",
    )
    google_maps_url: str = Field(
        default="", description="Link to the place on Google Maps"
    )


async def maps_request(
    api_key: SecretStr,
    method: str,
    url: str,
    *,
    params: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    field_mask: str = "",
) -> dict[str, Any]:
    """Call a Maps Platform REST API and return its JSON response."""
    headers = {"X-Goog-Api-Key": api_key.get_secret_value()}
    if field_mask:
        headers["X-Goog-FieldMask"] = field_mask
    response = await Requests(raise_for_status=False, retry_max_attempts=3).request(
        method,
        url,
        headers=headers,
        params=params,
        json=body,
        allow_redirects=False,
    )
    if not response.ok:
        raise maps_error(url, response)
    return response.json() if response.content else {}


def maps_error(url: str, response: Response) -> GoogleMapsError:
    """Turn a Maps Platform error response into a message the user can act on."""
    api = _API_NAMES.get(urlparse(url).hostname or "", "Google Maps API")
    body = response.json(fallback={}) if response.content else {}
    error = (body.get("error") if isinstance(body, dict) else None) or {}
    reasons = {detail.get("reason") for detail in error.get("details") or []}
    status = response.status
    for codes, template in _REASON_MESSAGES:
        if reasons & codes:
            return GoogleMapsError(template.format(api=api), status)
    if status == 401:
        return GoogleMapsError(_REASON_MESSAGES[0][1], status)
    if status == 429:
        return GoogleMapsError(
            f"This Maps API key has run out of {api} quota. Try again later, or "
            "raise the quota in the Google Cloud console.",
            status,
        )
    if status >= 500:
        return GoogleMapsError(
            f"The {api} had a temporary problem (HTTP {status}). Try again.", status
        )
    detail = error.get("message") or response.reason or "no details given"
    return GoogleMapsError(
        f"The {api} rejected the request (HTTP {status}): {detail}", status
    )


async def locate(
    api_key: SecretStr, text: str, *, language_code: str = ""
) -> tuple[MapsLocation, int]:
    """Coordinates for an address, place name, place ID or 'latitude,longitude'.

    Returns the location and the number of billable Places requests made: one
    Place Details Essentials lookup, unless the text was already coordinates.
    """
    if coordinates := parse_lat_lng(text):
        return MapsLocation(latitude=coordinates[0], longitude=coordinates[1]), 0
    place_id = parse_place_id(text) or await search_place_id(
        api_key, text, language_code=language_code
    )
    place = (
        await get_place(
            api_key, place_id, PLACE_ESSENTIALS_FIELDS, language_code=language_code
        )
        if place_id
        else {}
    )
    location = place.get("location")
    if not location:
        raise GoogleMapsError(
            f"Google Maps couldn't find '{text.strip()}'. Try a fuller address, "
            "or give coordinates as 'latitude,longitude'.",
            404,
        )
    return (
        MapsLocation(
            latitude=location.get("latitude", 0.0),
            longitude=location.get("longitude", 0.0),
            address=place.get("formattedAddress", ""),
            place_id=place.get("id", place_id),
        ),
        1,
    )


async def search_place_id(
    api_key: SecretStr,
    text: str,
    *,
    region_code: str = "",
    language_code: str = "",
    location_bias: dict[str, Any] | None = None,
    location_restriction: dict[str, Any] | None = None,
) -> str | None:
    """The place ID of the best match for a text query.

    Only the ID is requested, which Google doesn't charge for.
    """
    body: dict[str, Any] = {"textQuery": text, "pageSize": 1}
    if region_code:
        body["regionCode"] = region_code
    if language_code:
        body["languageCode"] = language_code
    if location_restriction:
        body["locationRestriction"] = location_restriction
    elif location_bias:
        body["locationBias"] = location_bias
    result = await maps_request(
        api_key,
        "POST",
        f"{PLACES_URL}/places:searchText",
        body=body,
        field_mask="places.id",
    )
    places = result.get("places") or []
    return places[0].get("id") if places else None


async def get_place(
    api_key: SecretStr,
    place_id: str,
    fields: str,
    *,
    region_code: str = "",
    language_code: str = "",
) -> dict[str, Any]:
    """A place's details. `fields` decides which Place Details price applies."""
    params = {}
    if region_code:
        params["regionCode"] = region_code
    if language_code:
        params["languageCode"] = language_code
    return await maps_request(
        api_key,
        "GET",
        f"{PLACES_URL}/places/{quote(place_id, safe='')}",
        params=params or None,
        field_mask=fields,
    )


def place_fields(place: dict[str, Any]) -> dict[str, Any]:
    """Map a Places API (New) place to MapsPlace fields."""
    location = place.get("location") or {}
    return {
        "place_id": place.get("id", ""),
        "name": dig(place, "displayName", "text") or "",
        "address": place.get("formattedAddress", ""),
        "latitude": location.get("latitude", 0.0) if location else None,
        "longitude": location.get("longitude", 0.0) if location else None,
        "types": place.get("types", []),
        "google_maps_url": place.get("googleMapsUri", ""),
    }


def parse_lat_lng(text: str) -> tuple[float, float] | None:
    """Read 'latitude,longitude' text, e.g. '48.8584, 2.2945'."""
    match = _LAT_LNG.fullmatch(text)
    if not match:
        return None
    latitude, longitude = float(match.group(1)), float(match.group(2))
    if abs(latitude) > 90 or abs(longitude) > 180:
        return None
    return latitude, longitude


def parse_place_id(text: str) -> str | None:
    """A place ID such as 'ChIJ...', or any place ID written as 'place_id:<id>'."""
    text = text.strip()
    if text.lower().startswith("place_id:"):
        return text[len("place_id:") :].strip() or None
    return text if _PLACE_ID.fullmatch(text) else None


def dig(data: dict[str, Any] | None, *keys: str) -> Any:
    """A nested value from an API response, or None if any level is missing."""
    for key in keys:
        data = (data or {}).get(key)
    return data
