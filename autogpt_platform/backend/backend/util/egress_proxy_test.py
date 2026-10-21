import asyncio
import socket

import pytest

from backend.util import egress_proxy
from backend.util.egress_proxy import egress_proxy_url, public_addresses


async def _socks_connect(
    proxy_url: str, host: str, port: int, address_type: int = 3, command: int = 1
) -> tuple[int, asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a SOCKS5 CONNECT through the proxy; return the reply code."""
    proxy_port = int(proxy_url.rsplit(":", 1)[1])
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    writer.write(b"\x05\x01\x00")
    assert await reader.readexactly(2) == b"\x05\x00"
    if address_type == 1:
        address = socket.inet_pton(socket.AF_INET, host)
    elif address_type == 4:
        address = socket.inet_pton(socket.AF_INET6, host)
    else:
        address = bytes([len(host)]) + host.encode()
    writer.write(
        bytes([5, command, 0, address_type]) + address + port.to_bytes(2, "big")
    )
    reply = await reader.readexactly(10)
    return reply[1], reader, writer


class _Server:
    """A loopback TCP server that echoes and counts its connections."""

    def __init__(self) -> None:
        self.connections = 0
        self.server: asyncio.Server | None = None

    async def __aenter__(self) -> "_Server":
        self.server = await asyncio.start_server(self._echo, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    @property
    def port(self) -> int:
        assert self.server is not None
        return self.server.sockets[0].getsockname()[1]

    async def _echo(self, reader, writer) -> None:
        self.connections += 1
        writer.write(await reader.read(1024))
        await writer.drain()
        writer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "localhost",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "192.0.2.1",
        "::1",
        "::",
        "::ffff:127.0.0.1",
        "fd00::1",
        "fe80::1",
        "fd00:ec2::254",
        # IPv6 forms that carry an IPv4 address a gateway may forward to
        "64:ff9b::a00:1",
        "64:ff9b::7f00:1",
        "64:ff9b::a9fe:a9fe",
        "64:ff9b:1::a00:1",
        "2002:a00:1::1",
        "2002:7f00:1::1",
        "2001:0:4136:e378:8000:63bf:f5ff:fffe",
        "::a00:1",
        "::7f00:1",
        # refused even when the embedded IPv4 is public
        "64:ff9b::808:808",
        "2002:808:808::1",
    ],
)
async def test_public_addresses_refuses_non_public_hosts(host):
    with pytest.raises(ValueError):
        await public_addresses(host)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host", ["8.8.8.8", "1.1.1.1", "::ffff:8.8.8.8", "2606:4700:4700::1111"]
)
async def test_public_addresses_allows_public_hosts(host):
    assert set(await public_addresses(host)) == {host}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host, address_type",
    [("127.0.0.1", 1), ("127.0.0.1", 3), ("localhost", 3), ("::ffff:127.0.0.1", 4)],
)
async def test_proxy_refuses_loopback_without_connecting(host, address_type):
    async with _Server() as server:
        code, _, writer = await _socks_connect(
            egress_proxy_url(), host, server.port, address_type
        )
        writer.close()

    assert code == egress_proxy._NOT_ALLOWED
    assert server.connections == 0


@pytest.mark.asyncio
async def test_proxy_dials_the_checked_address_not_the_name(monkeypatch):
    # "public.test" does not resolve; the relay only works because the proxy
    # connects to the address it checked instead of resolving the name again.
    async def vetted(host: str) -> list[str]:
        assert host == "public.test"
        return ["127.0.0.1"]

    monkeypatch.setattr(egress_proxy, "public_addresses", vetted)
    async with _Server() as server:
        code, reader, writer = await _socks_connect(
            egress_proxy_url(), "public.test", server.port
        )
        writer.write(b"ping")
        echoed = await reader.read(1024)
        writer.close()

    assert code == egress_proxy._SUCCEEDED
    assert echoed == b"ping"
    assert server.connections == 1


@pytest.mark.asyncio
async def test_proxy_never_dials_ipv6():
    # Passes every address check, as an address under an operator's own NAT64
    # prefix would, yet forwards to whatever IPv4 address it embeds.
    code, _, writer = await _socks_connect(
        egress_proxy_url(), "2600:1f18:64::a00:1", 80, address_type=4
    )
    writer.close()

    assert await public_addresses("2600:1f18:64::a00:1")
    assert code == egress_proxy._NOT_ALLOWED


@pytest.mark.asyncio
async def test_proxy_dials_the_ipv4_address_of_a_dual_stack_name(monkeypatch):
    async def vetted(host: str) -> list[str]:
        return ["2600:1f18:64::a00:1", "127.0.0.1"]

    monkeypatch.setattr(egress_proxy, "public_addresses", vetted)
    async with _Server() as server:
        code, reader, writer = await _socks_connect(
            egress_proxy_url(), "dual.test", server.port
        )
        writer.write(b"ping")
        echoed = await reader.read(1024)
        writer.close()

    assert code == egress_proxy._SUCCEEDED
    assert echoed == b"ping"


@pytest.mark.asyncio
async def test_proxy_refuses_a_client_without_a_supported_auth_method():
    proxy_port = int(egress_proxy_url().rsplit(":", 1)[1])
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    writer.write(b"\x05\x01\x02")  # offers username/password only
    reply = await reader.readexactly(2)
    writer.close()

    assert reply == bytes([5, egress_proxy._NO_ACCEPTABLE_METHODS])


@pytest.mark.asyncio
async def test_proxy_refuses_an_unknown_address_type():
    code, _, writer = await _socks_connect(
        egress_proxy_url(), "example.com", 80, address_type=9
    )
    writer.close()

    assert code == egress_proxy._ADDRESS_TYPE_NOT_SUPPORTED


@pytest.mark.asyncio
async def test_proxy_reports_an_unreachable_target(monkeypatch):
    async def vetted(host: str) -> list[str]:
        return ["127.0.0.1"]

    monkeypatch.setattr(egress_proxy, "public_addresses", vetted)
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    code, _, writer = await _socks_connect(egress_proxy_url(), "public.test", port)
    writer.close()

    assert code == egress_proxy._HOST_UNREACHABLE


@pytest.mark.asyncio
async def test_proxy_refuses_commands_other_than_connect():
    async with _Server() as server:
        code, _, writer = await _socks_connect(
            egress_proxy_url(), "127.0.0.1", server.port, address_type=1, command=3
        )
        writer.close()

    assert code == egress_proxy._COMMAND_NOT_SUPPORTED
    assert server.connections == 0


def test_proxy_url_is_stable_within_a_process():
    url = egress_proxy_url()
    assert url.startswith("socks5://127.0.0.1:")
    assert egress_proxy_url() == url
