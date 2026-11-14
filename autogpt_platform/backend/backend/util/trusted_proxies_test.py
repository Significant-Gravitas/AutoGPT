"""The client is named only by the proxies TRUSTED_PROXIES lists, the same way
for every reader."""

import ipaddress
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from backend.util.settings import Config
from backend.util.trusted_proxies import (
    client_address,
    parse_trusted_proxies,
    uvicorn_forwarded_allow_ips,
)

CLIENT = "203.0.113.7"
NGINX = "172.18.0.2"
HOST_PROXY = "172.18.0.5"
PROXIES = "172.18.0.0/16, 10.0.0.1"
# The single-container image's own parser reads the same vectors (test_entrypoint.py).
VECTORS = json.loads(
    (Path(__file__).with_name("trusted_proxies_vectors.json")).read_text()
)


@pytest.mark.parametrize(
    "peer, forwarded, expected",
    [
        # No proxy: an unlisted peer is the client, whatever it sent.
        (CLIENT, [], CLIENT),
        (CLIENT, ["6.6.6.6"], CLIENT),
        # One proxy, then two and three, each listed.
        (NGINX, [CLIENT], CLIENT),
        (NGINX, [f"{CLIENT}, {HOST_PROXY}"], CLIENT),
        ("10.0.0.1", [f"{CLIENT}, {HOST_PROXY}, {NGINX}"], CLIENT),
        # Entries the caller wrote sit left of the client and are never read.
        (NGINX, [f"6.6.6.6, {CLIENT}"], CLIENT),
        (NGINX, [f"6.6.6.6, 7.7.7.7, {CLIENT}, {HOST_PROXY}"], CLIENT),
        # A spoofed listed address left of the client does not move it either.
        (NGINX, [f"10.0.0.1, {CLIENT}"], CLIENT),
        # Lines a proxy added on its own read as one list.
        (NGINX, [CLIENT, HOST_PROXY], CLIENT),
        # A listed peer that forwards no header is the client.
        (NGINX, [], NGINX),
        (NGINX, [""], NGINX),
    ],
)
def test_the_client_is_the_first_unlisted_hop_from_the_right(
    peer: str, forwarded: list[str], expected: str
) -> None:
    assert client_address(peer, forwarded, PROXIES) == expected


@pytest.mark.parametrize(
    "proxies, peer, forwarded, expected",
    [
        (
            "2001:db8::/32",
            "2001:db8::1",
            "2001:db8:ffff::9, 2001:db9::7",
            "2001:db9::7",
        ),
        ("2001:db8::1", "2001:db8::1", f"[2001:db9::7]:4711, {CLIENT}", CLIENT),
        # As in uvicorn, an IPv4-mapped peer matches only a ::ffff: range.
        ("172.18.0.0/16", "::ffff:172.18.0.2", CLIENT, "::ffff:172.18.0.2"),
        # A port is not the address.
        ("::ffff:172.18.0.0/112", "::ffff:172.18.0.2", f"{CLIENT}:51000", CLIENT),
        # A host-bits CIDR means its network, as Better Auth reads it.
        ("172.18.0.9/16", NGINX, CLIENT, CLIENT),
    ],
)
def test_addresses_are_compared_as_addresses(
    proxies: str, peer: str, forwarded: str, expected: str
) -> None:
    assert client_address(peer, [forwarded], proxies) == expected


@pytest.mark.parametrize("value", ["nginx", "10.0.0.1/33", "10.0.0.0/8, 1.2.3"])
def test_an_entry_that_is_not_an_address_or_range_refuses_to_start(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The environment outranks init arguments in Config, so the value goes there.
    monkeypatch.setenv("TRUSTED_PROXIES", value)
    with pytest.raises(ValidationError, match="TRUSTED_PROXIES takes IP addresses"):
        Config()


@pytest.mark.parametrize("value", [",", " , ", " ,, "])
def test_a_value_that_lists_no_proxy_is_unset(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every reader then keeps its fallback, as the frontend and container do."""
    monkeypatch.setenv("TRUSTED_PROXIES", value)
    assert Config().trusted_proxies == ""


@pytest.mark.parametrize(
    "peer, forwarded",
    [
        (CLIENT, "6.6.6.6"),
        (NGINX, CLIENT),
        (NGINX, f"6.6.6.6, {CLIENT}, {HOST_PROXY}"),
        ("10.0.0.1", f"{CLIENT}, {HOST_PROXY}, {NGINX}"),
        (NGINX, f"{HOST_PROXY}, 172.18.9.9"),
        ("2001:db8::1", "2001:db9::7"),
        ("2001:db8::1", "2001:0db9:0:0:0:0:0:7"),
        # Entries that are not listed addresses: what uvicorn names, the walk names.
        (NGINX, "not-an-ip"),
        (NGINX, f"{CLIENT}, garbage, {HOST_PROXY}"),
        (NGINX, f"{CLIENT}, , {HOST_PROXY}"),
        (NGINX, f"{CLIENT}, [{HOST_PROXY}]junk"),
        (NGINX, f"{CLIENT}, [{HOST_PROXY}"),
        (NGINX, f"{CLIENT}, {HOST_PROXY}:notaport"),
        # A dual-stack socket's IPv4 peers, listed by an IPv4 range and a ::ffff: one.
        (f"::ffff:{NGINX}", CLIENT),
        ("::ffff:10.9.0.1", CLIENT),
    ],
)
async def test_uvicorn_names_the_same_client_and_a_second_walk_keeps_it(
    peer: str, forwarded: str
) -> None:
    """uvicorn rewrites `request.client` before the External API reads it, so
    the limiter re-walks from uvicorn's answer and must land on the same one."""
    proxies = f"{PROXIES}, 2001:db8::1, ::ffff:10.9.0.0/112"
    seen: list[str] = []

    async def app(scope, receive, send) -> None:
        seen.append(scope["client"][0])

    trusted_hosts = uvicorn_forwarded_allow_ips(proxies)
    assert trusted_hosts is not None
    middleware = ProxyHeadersMiddleware(app, trusted_hosts=trusted_hosts)
    scope = {
        "type": "http",
        "client": (peer, 1234),
        "headers": [(b"x-forwarded-for", forwarded.encode())],
    }
    await middleware(scope, AsyncMock(), AsyncMock())

    expected = client_address(peer, [forwarded], proxies)
    assert seen == [expected]
    assert client_address(seen[0], [forwarded], proxies) == expected


def test_uvicorn_follows_trusted_proxies_over_its_own_setting(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")

    with caplog.at_level(logging.WARNING):
        assert uvicorn_forwarded_allow_ips(PROXIES) == "172.18.0.0/16,10.0.0.1/32"

    assert "FORWARDED_ALLOW_IPS is ignored" in caplog.text


def test_without_trusted_proxies_uvicorn_keeps_forwarded_allow_ips(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.1")

    with caplog.at_level(logging.WARNING):
        assert uvicorn_forwarded_allow_ips("") is None

    assert "FORWARDED_ALLOW_IPS is deprecated" in caplog.text


def test_an_install_that_sets_neither_hears_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)

    with caplog.at_level(logging.WARNING):
        assert uvicorn_forwarded_allow_ips("") is None

    assert caplog.text == ""


def test_the_list_parses_once_per_value() -> None:
    assert parse_trusted_proxies(PROXIES) is parse_trusted_proxies(PROXIES)


@pytest.mark.parametrize("vector", VECTORS["accepted"], ids=lambda v: repr(v["value"]))
def test_parses_the_shared_vectors_the_container_parser_also_meets(vector) -> None:
    networks = parse_trusted_proxies(vector["value"])

    # Compared as networks: Python patch releases spell IPv4-mapped ranges differently.
    assert list(networks) == [ipaddress.ip_network(n) for n in vector["networks"]]


@pytest.mark.parametrize("vector", VECTORS["rejected"], ids=lambda v: repr(v["value"]))
def test_refuses_the_shared_vectors_the_container_parser_also_refuses(vector) -> None:
    with pytest.raises(ValueError) as refused:
        parse_trusted_proxies(vector["value"])

    assert str(refused.value) == (
        "TRUSTED_PROXIES takes IP addresses or CIDR ranges; "
        f"cannot parse: {vector['unparseable']}"
    )
