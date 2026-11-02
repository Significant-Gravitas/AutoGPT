"""The client is named only by the proxies TRUSTED_PROXIES lists, the same way
for every reader."""

import logging
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
    "forwarded, expected",
    [
        # What no listed proxy vouched for ends the walk at the nearest one.
        ("not-an-ip", NGINX),
        (f"{CLIENT}, garbage, {HOST_PROXY}", HOST_PROXY),
        # Only listed hops: the request started at the leftmost of them.
        (f"{HOST_PROXY}, 172.18.9.9", HOST_PROXY),
    ],
)
def test_a_header_with_no_unlisted_address_leaves_the_nearest_listed_hop(
    forwarded: str, expected: str
) -> None:
    assert client_address(NGINX, [forwarded], PROXIES) == expected


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
        # IPv4-mapped addresses match IPv4 ranges, and a port is not the address.
        ("172.18.0.0/16", "::ffff:172.18.0.2", f"{CLIENT}:51000", CLIENT),
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
    value: str,
) -> None:
    with pytest.raises(ValidationError, match="TRUSTED_PROXIES takes IP addresses"):
        Config(trusted_proxies=value)


@pytest.mark.parametrize(
    "peer, forwarded",
    [
        (CLIENT, "6.6.6.6"),
        (NGINX, CLIENT),
        (NGINX, f"6.6.6.6, {CLIENT}, {HOST_PROXY}"),
        ("10.0.0.1", f"{CLIENT}, {HOST_PROXY}, {NGINX}"),
        (NGINX, f"{HOST_PROXY}, 172.18.9.9"),
        ("2001:db8::1", "2001:db9::7"),
    ],
)
async def test_uvicorn_names_the_same_client_and_a_second_walk_keeps_it(
    peer: str, forwarded: str
) -> None:
    """uvicorn rewrites `request.client` before the External API reads it, so
    the limiter re-walks from uvicorn's answer and must land on the same one."""
    proxies = f"{PROXIES}, 2001:db8::1"
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
