"""Unit tests for reading Google Maps links, following short links, and
finding the place a link points at."""

from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest
from pydantic import SecretStr

from backend.blocks import _google_maps_api, _google_maps_links
from backend.blocks._google_maps_api import PLACE_PRO_FIELDS, GoogleMapsError
from backend.blocks._google_maps_links import (
    LinkError,
    LinkTarget,
    expand_link,
    find_link_place,
    parse_link,
)
from backend.util.request import Response

KEY = SecretStr("test-key")
PLACE = {"id": "ChIJLU7jZClu5kcR4PcOOO6p3I0", "displayName": {"text": "Eiffel Tower"}}
PLACE_URL = (
    "https://www.google.com/maps/place/Eiffel+Tower/@48.85837,2.29448,17z/"
    "data=!4m6!3m5!1s0x47e66e2964e34e2d:0x8ddca9ee380ef7e0!8m2"
    "!3d48.8583701!4d2.2944813!16zL20vMDJqODE"
)


def _redirect(location: str = "", status: int = 302) -> Response:
    headers = {"Location": location} if location else {}
    raw = MagicMock(status=status, headers=headers, reason="Reason")
    return Response(response=raw, url="https://example.invalid", body=b"")


class _FakeRequests:
    """Stands in for the Requests class: records calls, replies in order."""

    def __init__(self, *responses: Response):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, **kwargs: Any) -> "_FakeRequests":
        return self

    async def get(self, url: str, **kwargs: Any) -> Response:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class _FakeMaps:
    """Stands in for maps_request: records calls, replies in order."""

    def __init__(self, *responses: dict[str, Any] | Exception):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, api_key, method, url, *, params=None, body=None, field_mask=""
    ) -> dict[str, Any]:
        self.calls.append({"url": url, "body": body, "field_mask": field_mask})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_place_link_gives_the_name_and_the_pin():
    target = parse_link(PLACE_URL)
    assert target.name == "Eiffel Tower"
    assert (target.latitude, target.longitude) == (48.8583701, 2.2944813)
    assert target.exact is True
    assert target.place_id == ""


@pytest.mark.parametrize(
    "url, place_id, name, coordinates, exact",
    [
        (
            "https://www.google.com/maps/place/Caf%C3%A9+de+Flore/@48.854,2.3326,17z",
            "",
            "Café de Flore",
            (48.854, 2.3326),
            False,
        ),
        (
            "https://www.google.com/maps/search/?api=1&query=Eiffel+Tower"
            "&query_place_id=ChIJLU7jZClu5kcR4PcOOO6p3I0",
            "ChIJLU7jZClu5kcR4PcOOO6p3I0",
            "Eiffel Tower",
            None,
            False,
        ),
        (
            "https://maps.google.com/?q=place_id:ChIJLU7jZClu5kcR4PcOOO6p3I0",
            "ChIJLU7jZClu5kcR4PcOOO6p3I0",
            "",
            None,
            False,
        ),
        (
            "https://maps.google.com/?q=40.6892,-74.0445",
            "",
            "",
            (40.6892, -74.0445),
            True,
        ),
        (
            "https://www.google.co.uk/maps/@51.5007,-0.1246,15z",
            "",
            "",
            (51.5007, -0.1246),
            False,
        ),
        (
            "https://www.google.com/maps/place/48%C2%B051'30.1%22N+2%C2%B017'40.2%22E/"
            "@48.858,2.294,17z/data=!3m1!4b1!4m4!3m3!8m2!3d48.8583611!4d2.2945",
            "",
            "",
            (48.8583611, 2.2945),
            True,
        ),
    ],
)
def test_link_formats(
    url: str,
    place_id: str,
    name: str,
    coordinates: tuple[float, float] | None,
    exact: bool,
):
    target = parse_link(url)
    assert target.place_id == place_id
    assert target.name == name
    found = None if target.latitude is None else (target.latitude, target.longitude)
    assert found == coordinates
    assert target.exact is exact


@pytest.mark.parametrize(
    "url, reason",
    [
        ("https://www.google.com/maps/dir/Paris/Lyon", "directions link"),
        ("https://maps.google.com/?cid=10222232094831998944", "Google's internal ID"),
        ("https://www.google.com/maps", "doesn't point to a place"),
    ],
)
def test_links_without_one_place(url: str, reason: str):
    with pytest.raises(LinkError, match=reason):
        parse_link(url)


@pytest.mark.asyncio
async def test_long_links_are_not_fetched(monkeypatch):
    fake = _FakeRequests()
    monkeypatch.setattr(_google_maps_links, "Requests", fake)
    assert await expand_link(f" {PLACE_URL} ") == PLACE_URL
    assert (
        await expand_link("maps.google.com/?q=Louvre")
        == "https://maps.google.com/?q=Louvre"
    )
    assert fake.calls == []


@pytest.mark.asyncio
async def test_other_sites_are_refused(monkeypatch):
    fake = _FakeRequests()
    monkeypatch.setattr(_google_maps_links, "Requests", fake)
    with pytest.raises(LinkError, match="isn't a Google Maps link"):
        await expand_link("https://example.com/maps/place/Eiffel+Tower")
    with pytest.raises(LinkError, match="isn't a Google Maps link"):
        await expand_link("https://goo.gl/abc123")
    assert fake.calls == []


@pytest.mark.asyncio
async def test_short_links_are_followed_one_hop_at_a_time(monkeypatch):
    fake = _FakeRequests(
        _redirect("https://maps.app.goo.gl/Second"), _redirect(PLACE_URL)
    )
    monkeypatch.setattr(_google_maps_links, "Requests", fake)
    assert await expand_link("https://maps.app.goo.gl/First?g_st=ic") == PLACE_URL
    assert [url for url, _ in fake.calls] == [
        "https://maps.app.goo.gl/First?g_st=ic",
        "https://maps.app.goo.gl/Second",
    ]
    assert all(kwargs["allow_redirects"] is False for _, kwargs in fake.calls)


@pytest.mark.asyncio
async def test_consent_redirects_are_unwrapped(monkeypatch):
    consent = "https://consent.google.com/ml?continue=" + quote(PLACE_URL, safe="")
    monkeypatch.setattr(
        _google_maps_links, "Requests", _FakeRequests(_redirect(consent))
    )
    assert await expand_link("https://maps.app.goo.gl/Abc") == PLACE_URL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, reason",
    [
        (_redirect(status=404), "expired or doesn't exist"),
        (_redirect(status=200), "didn't redirect"),
        (_redirect("https://example.com/"), "doesn't lead to a Google Maps page"),
    ],
)
async def test_short_link_failures(monkeypatch, response: Response, reason: str):
    monkeypatch.setattr(_google_maps_links, "Requests", _FakeRequests(response))
    with pytest.raises(LinkError, match=reason):
        await expand_link("https://maps.app.goo.gl/Abc")


@pytest.mark.asyncio
async def test_short_link_redirect_loops_stop(monkeypatch):
    loop = [_redirect("https://maps.app.goo.gl/Abc") for _ in range(5)]
    monkeypatch.setattr(_google_maps_links, "Requests", _FakeRequests(*loop))
    with pytest.raises(LinkError, match="too many times"):
        await expand_link("https://maps.app.goo.gl/Abc")


@pytest.mark.asyncio
async def test_pinned_names_are_searched_only_around_the_pin(monkeypatch):
    fake = _FakeMaps({"places": [{"id": "ChIJPlace"}]}, PLACE)
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    assert await find_link_place(KEY, parse_link(PLACE_URL)) == PLACE
    body = fake.calls[0]["body"]
    assert body["textQuery"] == "Eiffel Tower"
    box = body["locationRestriction"]["rectangle"]
    assert box["low"]["latitude"] == pytest.approx(48.8483701)
    assert box["high"]["longitude"] == pytest.approx(2.3044813)
    assert "locationBias" not in body
    assert fake.calls[1]["url"].endswith("/places/ChIJPlace")
    assert fake.calls[1]["field_mask"] == PLACE_PRO_FIELDS


@pytest.mark.asyncio
async def test_names_near_a_map_centre_are_biased_not_restricted(monkeypatch):
    fake = _FakeMaps({})
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    target = LinkTarget(expanded_url="u", name="Louvre", latitude=48.86, longitude=2.33)
    assert await find_link_place(KEY, target) is None
    body = fake.calls[0]["body"]
    assert body["locationBias"]["circle"]["center"] == {
        "latitude": 48.86,
        "longitude": 2.33,
    }
    assert "locationRestriction" not in body


@pytest.mark.asyncio
async def test_pin_box_wraps_at_the_antimeridian(monkeypatch):
    fake = _FakeMaps({})
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    target = LinkTarget(
        expanded_url="u", name="Taveuni", latitude=-16.8, longitude=179.995, exact=True
    )
    await find_link_place(KEY, target)
    box = fake.calls[0]["body"]["locationRestriction"]["rectangle"]
    assert box["low"]["longitude"] == pytest.approx(179.985)
    assert box["high"]["longitude"] == pytest.approx(-179.995)


@pytest.mark.asyncio
async def test_unknown_place_ids_fail_the_link(monkeypatch):
    fake = _FakeMaps(GoogleMapsError("Not a valid place ID", 400))
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    with pytest.raises(LinkError, match="doesn't recognise the place ID"):
        await find_link_place(KEY, LinkTarget(expanded_url="u", place_id="bad"))


@pytest.mark.asyncio
async def test_other_api_errors_are_not_link_failures(monkeypatch):
    fake = _FakeMaps(GoogleMapsError("The Places API (New) isn't enabled", 403))
    monkeypatch.setattr(_google_maps_api, "maps_request", fake)
    with pytest.raises(GoogleMapsError, match="isn't enabled"):
        await find_link_place(KEY, LinkTarget(expanded_url="u", place_id="ChIJx"))
