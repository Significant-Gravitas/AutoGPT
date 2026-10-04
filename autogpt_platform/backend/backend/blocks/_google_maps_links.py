"""Expand and read Google Maps links for the resolve-links block."""

import re
from typing import Any
from urllib.parse import ParseResult, parse_qs, unquote_plus, urljoin, urlparse

import aiohttp
from pydantic import BaseModel, SecretStr

from backend.blocks._google_maps_api import (
    PLACE_PRO_FIELDS,
    GoogleMapsError,
    get_place,
    parse_lat_lng,
    search_place_id,
)
from backend.util.request import Requests

_MAX_HOPS = 5
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_GOOGLE_HOST = re.compile(r"(?:www\.)?google(?:\.[a-z]{2,3}){1,2}")
_MAPS_HOST = re.compile(r"maps\.google(?:\.[a-z]{2,3}){1,2}")
# Place links carry the place's own pin as !3d<lat>!4d<lng> in their data part,
# and the map's centre as @<lat>,<lng>.
_PIN = re.compile(r"!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)")
_CENTRE = re.compile(r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")
_NAMED = re.compile(r"/maps/(?:place|search)/([^/]+)")
# How far from a link's pin, in degrees (about 1 km), a name may match.
_PIN_MARGIN = 0.01
_CENTRE_RADIUS_METERS = 5000.0


class LinkError(Exception):
    """A link that can't be resolved, with the reason to show the user."""


class LinkTarget(BaseModel):
    """What a Google Maps link points at."""

    expanded_url: str
    place_id: str = ""
    name: str = ""
    latitude: float | None = None
    longitude: float | None = None
    # True when the coordinates are the place's own pin, not the map's centre.
    exact: bool = False


async def expand_link(url: str) -> str:
    """The full Google Maps URL behind a link, following short-link redirects.

    Only maps.app.goo.gl and goo.gl/maps links are fetched, one redirect at a
    time. The Google Maps page itself is never fetched.
    """
    url = url.strip()
    if "://" not in url:
        url = f"https://{url}"
    if is_maps_url(urlparse(url)):
        return url
    if not is_short_link(urlparse(url)):
        raise LinkError("This isn't a Google Maps link.")
    for _ in range(_MAX_HOPS):
        url = await _follow_redirect(url)
        if is_maps_url(urlparse(url)):
            return url
        if not is_short_link(urlparse(url)):
            raise LinkError("The short link doesn't lead to a Google Maps page.")
    raise LinkError("The short link redirected too many times.")


def parse_link(url: str) -> LinkTarget:
    """Read the place ID, name and coordinates that a Google Maps URL carries."""
    parsed = urlparse(url)
    if parsed.path.startswith("/maps/dir"):
        raise LinkError("This is a directions link, not a link to one place.")
    params = {key: values[0] for key, values in parse_qs(parsed.query).items()}
    query = params.get("q") or params.get("query") or ""
    target = LinkTarget(expanded_url=url, place_id=params.get("query_place_id", ""))
    if query.lower().startswith("place_id:"):
        target.place_id, query = query[len("place_id:") :].strip(), ""
    named = _NAMED.search(parsed.path)
    name = unquote_plus(named.group(1)) if named else query.strip()
    if pin := parse_lat_lng(name):
        target.latitude, target.longitude = pin
        target.exact = True
    elif "°" not in name:
        target.name = name
    if target.latitude is None:
        _read_coordinates(target, parsed.path, params)
    if not (target.place_id or target.name or target.latitude is not None):
        raise LinkError(_no_place_reason(url, params))
    return target


async def find_link_place(
    api_key: SecretStr, target: LinkTarget
) -> dict[str, Any] | None:
    """Place details (name, address, types, link) for a link's place.

    Returns None when a name search finds nothing near the link's location.
    """
    place_id = target.place_id or await search_place_id(
        api_key,
        target.name,
        location_restriction=_pin_box(target) if target.exact else None,
        location_bias=_centre_circle(target) if not target.exact else None,
    )
    if not place_id:
        return None
    try:
        return await get_place(api_key, place_id, PLACE_PRO_FIELDS)
    except GoogleMapsError as e:
        if e.status in (400, 404):
            raise LinkError(
                "Google Maps doesn't recognise the place ID in this link."
            ) from e
        raise


def is_short_link(parsed: ParseResult) -> bool:
    host = parsed.hostname or ""
    return parsed.scheme in ("http", "https") and (
        host == "maps.app.goo.gl"
        or (host == "goo.gl" and parsed.path.startswith("/maps"))
    )


def is_maps_url(parsed: ParseResult) -> bool:
    host = parsed.hostname or ""
    if parsed.scheme not in ("http", "https"):
        return False
    if _MAPS_HOST.fullmatch(host):
        return True
    return bool(_GOOGLE_HOST.fullmatch(host)) and (
        parsed.path == "/maps" or parsed.path.startswith("/maps/")
    )


async def _follow_redirect(url: str) -> str:
    try:
        response = await Requests(raise_for_status=False, retry_max_attempts=2).get(
            url, allow_redirects=False
        )
    except (aiohttp.ClientError, TimeoutError, ValueError) as e:
        raise LinkError(f"Couldn't open the short link: {e}") from e
    if response.status == 404:
        raise LinkError("The short link has expired or doesn't exist.")
    location = response.headers.get("Location")
    if response.status not in _REDIRECT_STATUSES or not location:
        raise LinkError("The short link didn't redirect to Google Maps.")
    next_url = urlparse(urljoin(url, location))
    # Google's cookie-consent page carries the real destination in `continue`.
    if next_url.hostname == "consent.google.com":
        return parse_qs(next_url.query).get("continue", [next_url.geturl()])[0]
    return next_url.geturl()


def _read_coordinates(target: LinkTarget, path: str, params: dict[str, str]) -> None:
    if pins := _PIN.findall(target.expanded_url):
        target.latitude, target.longitude = float(pins[-1][0]), float(pins[-1][1])
        target.exact = True
    elif centre := parse_lat_lng(params.get("ll", "")):
        target.latitude, target.longitude = centre
    elif match := _CENTRE.search(path):
        target.latitude, target.longitude = float(match[1]), float(match[2])


def _no_place_reason(url: str, params: dict[str, str]) -> str:
    if "cid" in params or "ftid" in params or "!1s0x" in url:
        return (
            "This link names the place only by Google's internal ID (CID), "
            "which the Places API can't look up."
        )
    return "This link doesn't point to a place."


def _pin_box(target: LinkTarget) -> dict[str, Any]:
    latitude, longitude = target.latitude or 0.0, target.longitude or 0.0
    return {
        "rectangle": {
            "low": {
                "latitude": max(-90.0, latitude - _PIN_MARGIN),
                "longitude": _wrap(longitude - _PIN_MARGIN),
            },
            "high": {
                "latitude": min(90.0, latitude + _PIN_MARGIN),
                "longitude": _wrap(longitude + _PIN_MARGIN),
            },
        }
    }


def _centre_circle(target: LinkTarget) -> dict[str, Any] | None:
    if target.latitude is None or target.longitude is None:
        return None
    return {
        "circle": {
            "center": {"latitude": target.latitude, "longitude": target.longitude},
            "radius": _CENTRE_RADIUS_METERS,
        }
    }


def _wrap(longitude: float) -> float:
    return (longitude + 180.0) % 360.0 - 180.0
