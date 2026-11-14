import asyncio
import ipaddress
import os
import socket
from pathlib import Path

HANDSHAKE_SECONDS = 10
# Every tunnel is cut after this long; the browser reconnects if it must.
TUNNEL_SECONDS = 300


def destination(authority: str, allowed_hosts: set[str]) -> str:
    host, separator, port = authority.rpartition(":")
    if (
        separator != ":"
        or port != "443"
        or not host
        or any(c in host for c in "@/\\?#[]%")
    ):
        raise ValueError("Only named HTTPS destinations are supported")
    host = host.lower().encode("idna").decode("ascii")
    if host not in allowed_hosts or host.endswith("."):
        raise ValueError("Destination not approved")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host
    raise ValueError("IP literals are not permitted")


def public_addresses(values: list[str]) -> list[str]:
    addresses = [ipaddress.ip_address(value) for value in values]
    if not addresses or any(not public_address(address) for address in addresses):
        raise ValueError("Destination resolves to a non-public address")
    return list(dict.fromkeys(values))


def public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if not address.is_global:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        return address in ipaddress.IPv6Network("2000::/3") and not (
            address.sixtofour or address.teredo or address.ipv4_mapped
        )
    return True


async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(65536):
        writer.write(data)
        await writer.drain()


async def tunnel(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, allowed_hosts: set[str]
) -> None:
    upstream_writer: asyncio.StreamWriter | None = None
    established = False
    try:
        async with asyncio.timeout(TUNNEL_SECONDS):
            async with asyncio.timeout(HANDSHAKE_SECONDS):
                header = await reader.readuntil(b"\r\n\r\n")
                if len(header) > 8192:
                    raise ValueError("Request too large")
                method, authority, version = (
                    header.split(b"\r\n", 1)[0].decode("ascii").split(" ")
                )
                if method != "CONNECT" or version not in {"HTTP/1.0", "HTTP/1.1"}:
                    raise ValueError("Only HTTPS tunnels are supported")
                host = destination(authority, allowed_hosts)
                answers = await asyncio.get_running_loop().getaddrinfo(
                    host, 443, type=socket.SOCK_STREAM
                )
                addresses = public_addresses([str(answer[4][0]) for answer in answers])
                upstream_reader, upstream_writer = await asyncio.open_connection(
                    addresses[0], 443
                )
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
                established = True
            tasks = [
                asyncio.create_task(relay(reader, upstream_writer)),
                asyncio.create_task(relay(upstream_reader, writer)),
            ]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    except Exception:
        # Once the tunnel is open the stream is the client's TLS, and closing
        # it is the only safe ending; a refusal goes only where none opened.
        if not established:
            writer.write(
                b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
    finally:
        writer.close()
        if upstream_writer:
            upstream_writer.close()


async def main() -> None:
    hosts = {
        line.strip().lower()
        for line in Path(os.environ["CHECKOUT_EGRESS_HOSTS_FILE"])
        .read_text()
        .splitlines()
        if line.strip() and not line.startswith("#")
    }
    if not hosts or "*" in "".join(hosts):
        raise ValueError("An exact destination allowlist is required")
    server = await asyncio.start_server(
        lambda r, w: tunnel(r, w, hosts), "0.0.0.0", 3128, limit=8192
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
