import asyncio
import socket
from contextlib import asynccontextmanager

import pytest

from backend.util.link_checkout import egress
from backend.util.link_checkout.egress import destination, public_addresses


@pytest.mark.parametrize(
    "authority",
    [
        "shop.example:80",
        "shop.example:22",
        "evil.example:443",
        "shop.example.:443",
        "shop.example@evil.example:443",
        "127.0.0.1:443",
        "[::1]:443",
        "shop.example/path:443",
        "shop.example%00:443",
    ],
)
def test_only_explicit_https_hosts_can_receive_browser_traffic(authority):
    with pytest.raises(ValueError):
        destination(authority, {"shop.example", "127.0.0.1"})


@pytest.mark.parametrize(
    "addresses",
    [
        ["127.0.0.1"],
        ["10.0.0.1"],
        ["169.254.169.254"],
        ["::1"],
        ["fc00::1"],
        ["::ffff:127.0.0.1"],
        ["64:ff9b::7f00:1"],
        ["2002:7f00:1::"],
        ["2001:0:4136:e378:8000:63bf:3fff:fdd2"],
        ["93.184.216.34", "192.168.1.1"],
        [],
    ],
)
def test_any_private_dns_answer_denies_the_entire_connection(addresses):
    with pytest.raises(ValueError):
        public_addresses(addresses)


def test_public_address_is_resolved_once_and_pinned_for_connect():
    assert destination("shop.example:443", {"shop.example"}) == "shop.example"
    assert public_addresses(["93.184.216.34", "93.184.216.34"]) == ["93.184.216.34"]


async def _pair():
    left, right = socket.socketpair()
    return (
        await asyncio.open_connection(sock=left),
        await asyncio.open_connection(sock=right),
    )


@asynccontextmanager
async def running_tunnel(monkeypatch):
    """A tunnel whose client and upstream are local socket pairs."""
    (tunnel_reader, tunnel_writer), client = await _pair()
    upstream_for_tunnel, upstream = await _pair()

    async def resolve(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    async def connect(host, port):
        assert (host, port) == ("93.184.216.34", 443)
        return upstream_for_tunnel

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    monkeypatch.setattr(egress.asyncio, "open_connection", connect)
    task = asyncio.create_task(
        egress.tunnel(tunnel_reader, tunnel_writer, {"shop.example"})
    )
    try:
        yield client, upstream
    finally:
        await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_an_open_tunnel_ends_by_closing_not_with_a_refusal(monkeypatch):
    monkeypatch.setattr(egress, "TUNNEL_SECONDS", 0.5)
    async with running_tunnel(monkeypatch) as (client, upstream):
        client_reader, client_writer = client
        client_writer.write(b"CONNECT shop.example:443 HTTP/1.1\r\n\r\n")
        await client_writer.drain()
        assert await client_reader.readuntil(b"\r\n\r\n") == (
            b"HTTP/1.1 200 Connection Established\r\n\r\n"
        )
        client_writer.write(b"tls-bytes")
        await client_writer.drain()
        assert await upstream[0].readexactly(9) == b"tls-bytes"

        # The deadline passes with the tunnel open: nothing may be written
        # into what is now the client's TLS stream.
        assert await asyncio.wait_for(client_reader.read(), 5) == b""


@pytest.mark.asyncio
async def test_a_refused_destination_still_gets_a_403(monkeypatch):
    async with running_tunnel(monkeypatch) as (client, _):
        client_reader, client_writer = client
        client_writer.write(b"CONNECT evil.example:443 HTTP/1.1\r\n\r\n")
        await client_writer.drain()

        refusal = await asyncio.wait_for(client_reader.read(), 5)

    assert refusal.startswith(b"HTTP/1.1 403 Forbidden")
