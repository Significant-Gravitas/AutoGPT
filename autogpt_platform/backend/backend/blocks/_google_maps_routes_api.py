"""Routes API models, requests and response parsing for the directions block."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator

from pydantic import BaseModel, Field, SecretStr

from backend.blocks._google_maps_api import (
    ROUTES_URL,
    UnitsSystem,
    dig,
    maps_request,
    parse_lat_lng,
    parse_place_id,
)


class TravelMode(str, Enum):
    DRIVE = "drive"
    WALK = "walk"
    BICYCLE = "bicycle"
    TRANSIT = "transit"
    TWO_WHEELER = "two_wheeler"


TRAFFIC_MODES = {TravelMode.DRIVE, TravelMode.TWO_WHEELER}
MODE_WORDS = {
    TravelMode.DRIVE: "driving",
    TravelMode.WALK: "walking",
    TravelMode.BICYCLE: "cycling",
    TravelMode.TRANSIT: "transit",
    TravelMode.TWO_WHEELER: "two-wheeler",
}
_ROUTE_FIELDS = [
    "routes.distanceMeters",
    "routes.duration",
    "routes.description",
    "routes.warnings",
    "routes.localizedValues",
]
_STEP_FIELDS = [
    "routes.legs.steps.distanceMeters",
    "routes.legs.steps.staticDuration",
    "routes.legs.steps.navigationInstruction",
    "routes.legs.steps.localizedValues",
    "routes.legs.steps.travelMode",
    "routes.legs.steps.transitDetails",
]


class TransitDetails(BaseModel):
    """The public transport part of a transit step."""

    line: str = Field(default="", description="Line name, e.g. 'M1' or 'Red Line'")
    vehicle: str = Field(
        default="", description="Kind of vehicle, e.g. Bus, Subway or Train"
    )
    headsign: str = Field(default="", description="Destination shown on the vehicle")
    departure_stop: str = Field(default="", description="Stop to get on at")
    arrival_stop: str = Field(default="", description="Stop to get off at")
    departure_time: str = Field(
        default="", description="Local departure time, e.g. '8:05 AM'"
    )
    arrival_time: str = Field(default="", description="Local arrival time")
    stop_count: int = Field(default=0, description="Number of stops to ride")


class RouteStep(BaseModel):
    """One step of a route."""

    instruction: str = Field(
        default="", description="What to do, e.g. 'Turn left onto Rue de Rivoli'"
    )
    maneuver: str = Field(
        default="", description="Google's maneuver code, e.g. TURN_LEFT"
    )
    distance_meters: int = Field(default=0, description="Length of the step in meters")
    distance_text: str = Field(default="", description="Length as text, e.g. '350 m'")
    duration_seconds: int = Field(
        default=0, description="Time for the step in seconds, without traffic"
    )
    duration_text: str = Field(default="", description="Time as text, e.g. '2 mins'")
    travel_mode: str = Field(
        default="",
        description="How this step is travelled, e.g. DRIVE, WALK or TRANSIT",
    )
    transit: TransitDetails | None = Field(
        default=None, description="Line, stops and times, for transit steps"
    )


def build_route_request(
    *,
    origin: str,
    destination: str,
    travel_mode: TravelMode,
    use_live_traffic: bool,
    units: UnitsSystem,
    language_code: str = "",
    departure_time: datetime | None = None,
) -> dict[str, Any]:
    """The computeRoutes request body."""
    body: dict[str, Any] = {
        "origin": to_waypoint(origin),
        "destination": to_waypoint(destination),
        "travelMode": travel_mode.value.upper(),
        "units": units.value.upper(),
    }
    # Only drive and two-wheeler routes take a routing preference. Google
    # bills TRAFFIC_AWARE at its higher Pro rate.
    if travel_mode in TRAFFIC_MODES:
        body["routingPreference"] = (
            "TRAFFIC_AWARE" if use_live_traffic else "TRAFFIC_UNAWARE"
        )
    if language_code:
        body["languageCode"] = language_code
    if departure_time and (travel_mode == TravelMode.TRANSIT or use_live_traffic):
        if departure_time.tzinfo is None:
            departure_time = departure_time.replace(tzinfo=timezone.utc)
        body["departureTime"] = (
            departure_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        )
    return body


def to_waypoint(text: str) -> dict[str, Any]:
    """A Routes API waypoint from coordinates, a place ID, or an address or name."""
    if coordinates := parse_lat_lng(text):
        latitude, longitude = coordinates
        return {"location": {"latLng": {"latitude": latitude, "longitude": longitude}}}
    if place_id := parse_place_id(text):
        return {"placeId": place_id}
    return {"address": text.strip()}


async def compute_route(
    api_key: SecretStr, body: dict[str, Any], *, include_steps: bool
) -> dict[str, Any]:
    fields = _ROUTE_FIELDS + (_STEP_FIELDS if include_steps else [])
    return await maps_request(
        api_key, "POST", ROUTES_URL, body=body, field_mask=",".join(fields)
    )


def route_outputs(
    route: dict[str, Any], *, include_steps: bool
) -> Iterator[tuple[str, Any]]:
    """The directions block's outputs for one route, in output order."""
    yield "summary", route_summary(route)
    yield "distance_meters", route.get("distanceMeters", 0)
    yield "distance_text", dig(route, "localizedValues", "distance", "text") or ""
    yield "duration_seconds", _seconds(route.get("duration"))
    yield "duration_text", dig(route, "localizedValues", "duration", "text") or ""
    if include_steps:
        steps = [
            to_route_step(step)
            for leg in route.get("legs") or []
            for step in leg.get("steps") or []
        ]
        yield "steps", steps
        yield from (("step", step) for step in steps)
    yield "warnings", route.get("warnings") or []


def route_summary(route: dict[str, Any]) -> str:
    duration = (
        dig(route, "localizedValues", "duration", "text")
        or f"{round(_seconds(route.get('duration')) / 60)} min"
    )
    distance = (
        dig(route, "localizedValues", "distance", "text")
        or f"{route.get('distanceMeters', 0) / 1000:.1f} km"
    )
    summary = f"{duration} ({distance})"
    if description := route.get("description"):
        summary += f" via {description}"
    return summary


def to_route_step(step: dict[str, Any]) -> RouteStep:
    transit = step.get("transitDetails")
    return RouteStep(
        instruction=dig(step, "navigationInstruction", "instructions") or "",
        maneuver=dig(step, "navigationInstruction", "maneuver") or "",
        distance_meters=step.get("distanceMeters", 0),
        distance_text=dig(step, "localizedValues", "distance", "text") or "",
        duration_seconds=_seconds(step.get("staticDuration")),
        duration_text=dig(step, "localizedValues", "staticDuration", "text") or "",
        travel_mode=step.get("travelMode", ""),
        transit=to_transit_details(transit) if transit else None,
    )


def to_transit_details(details: dict[str, Any]) -> TransitDetails:
    line = details.get("transitLine") or {}
    return TransitDetails(
        line=line.get("nameShort") or line.get("name") or "",
        vehicle=dig(line, "vehicle", "name", "text") or "",
        headsign=details.get("headsign", ""),
        departure_stop=dig(details, "stopDetails", "departureStop", "name") or "",
        arrival_stop=dig(details, "stopDetails", "arrivalStop", "name") or "",
        departure_time=dig(details, "localizedValues", "departureTime", "time", "text")
        or "",
        arrival_time=dig(details, "localizedValues", "arrivalTime", "time", "text")
        or "",
        stop_count=details.get("stopCount", 0),
    )


def _seconds(duration: str | None) -> int:
    """Seconds from a Google duration string such as '868s'."""
    return round(float(duration.rstrip("s"))) if duration else 0
