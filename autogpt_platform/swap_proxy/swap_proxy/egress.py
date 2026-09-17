"""The egress guard: a box must not reach private address space through us.

The proxy sits inside the platform's network and dials out on a box's behalf,
so without this a box could ask it for the metadata server, the database or a
neighbouring service.  Ported from spark-vm's ``server_connect`` guard.

The destination is resolved *before* the upstream connect, a refused address
never gets even a SYN, and an allowed one is pinned to the address that was
checked, so the check and the connect use the same answer: no DNS-rebind race.

One departure from spark-vm: a DNS failure is refused, not waved through.
There the connect "fails on its own anyway"; here mitmproxy would resolve the
name again itself, and a name that fails once and answers with a private
address a moment later is exactly the race the pin exists to close.
"""

import asyncio
import ipaddress
import socket
import time
from dataclasses import dataclass
from typing import Optional, Union

from swap_proxy.swap import host_in_list

IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]
IPNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]

# RFC 1918, loopback, link-local (cloud metadata lives there), CGNAT and other
# carrier space, 0.0.0.0/8, IETF assignments, benchmarking and reserved space,
# multicast, and ULA / link-local / multicast v6.
PRIVATE_NETS: tuple[IPNetwork, ...] = tuple(
    ipaddress.ip_network(c)
    for c in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    )
)
_DNS_TTL = 60
_DNS_CACHE_MAX = 4096


@dataclass(frozen=True)
class Verdict:
    """Where to connect, or why not.  Exactly one of the two is set."""

    ip: Optional[str] = None
    refused: Optional[str] = None
    refused_ip: Optional[str] = None


def normalize_ip(ip: str) -> Optional[IPAddress]:
    """Parse a literal, unwrapping IPv4-mapped IPv6: ``::ffff:127.0.0.1``
    reaches localhost and must be judged as 127.0.0.1."""
    try:
        addr = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return None
    mapped = getattr(addr, "ipv4_mapped", None)
    return mapped if mapped is not None else addr


def is_private(addr: IPAddress) -> bool:
    return addr.is_unspecified or any(addr in net for net in PRIVATE_NETS)


def parse_allow(entries: list[str]) -> tuple[list[str], list[IPNetwork]]:
    """Hostnames (exact or leading-dot) and CIDR or address literals."""
    hosts: list[str] = []
    nets: list[IPNetwork] = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            hosts.append(entry.lower())
    return hosts, nets


class EgressGuard:
    """Default deny for private space; *allow* is for tests and for the rare
    internal host a deployment means its boxes to reach."""

    def __init__(self, allow: Optional[list[str]] = None):
        self.allow_hosts, self.allow_nets = parse_allow(allow or [])
        self._dns: dict[str, tuple[float, Optional[list[str]]]] = {}

    async def check(self, host: str) -> Verdict:
        ips = await self._resolve(host)
        if not ips:
            return Verdict(refused="unresolvable")
        refused_ip = None
        for ip in ips:
            addr = normalize_ip(ip)
            if addr is None:
                continue
            if is_private(addr) and not self._allowed(host, addr):
                refused_ip = refused_ip or str(addr)
                continue
            return Verdict(ip=str(addr))
        return Verdict(refused="private-range", refused_ip=refused_ip)

    def _allowed(self, host: str, addr: IPAddress) -> bool:
        if host_in_list(host, self.allow_hosts):
            return True
        return any(addr in net for net in self.allow_nets)

    async def _resolve(self, host: str) -> Optional[list[str]]:
        """Through the loop's resolver: this runs on mitmproxy's event loop,
        and a slow lookup must not stall every other connection."""
        now = time.monotonic()
        cached = self._dns.get(host)
        if cached and cached[0] > now:
            return cached[1]
        ips: Optional[list[str]]
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                host, None, type=socket.SOCK_STREAM
            )
        except (socket.gaierror, UnicodeError, OSError):
            ips = None
        else:
            # IPv4 first: pinning one address gives up the dialler's fallback
            # from an unroutable AAAA to the A record, and the networks this
            # runs in are more often v4-only than v6-only.
            ordered = sorted(infos, key=lambda info: info[0] != socket.AF_INET)
            ips = list(dict.fromkeys(str(info[4][0]) for info in ordered))
        if len(self._dns) >= _DNS_CACHE_MAX:
            self._dns.clear()  # only a latency optimisation
        self._dns[host] = (now + _DNS_TTL, ips)
        return ips
