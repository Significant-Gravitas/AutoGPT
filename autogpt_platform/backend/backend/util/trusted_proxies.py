"""The client behind the proxies listed in `TRUSTED_PROXIES`.

One list drives every reader of X-Forwarded-For: uvicorn's proxy headers (and so
`request.client`), the External API's per-address limits, and the frontend.
"""

import functools
import ipaddress
import logging
import os
from collections.abc import Iterable

logger = logging.getLogger(__name__)

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


def client_address(peer: str, forwarded_for: Iterable[str], proxies: str) -> str:
    """The first X-Forwarded-For address from the right that `proxies` does not list.

    Only a listed peer may name the client, so an unlisted one is the client
    itself. An entry that is not an address, or a header of listed addresses
    only, leaves the client at the nearest listed hop: the walk never trusts a
    value no listed proxy vouched for.
    """
    networks = parse_trusted_proxies(proxies)
    peer_address = _address(peer)
    if peer_address is None or not _is_listed(peer_address, networks):
        return peer

    nearest = str(peer_address)
    for entry in reversed(forwarded_entries(forwarded_for)):
        address = _address(entry)
        if address is None:
            return nearest
        if not _is_listed(address, networks):
            return str(address)
        nearest = str(address)
    return nearest


@functools.cache
def parse_trusted_proxies(value: str) -> tuple[IPNetwork, ...]:
    """`value`'s comma-separated addresses and CIDR ranges, refusing any other entry."""
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    networks: list[IPNetwork] = []
    invalid: list[str] = []
    for entry in entries:
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            invalid.append(entry)
    if invalid:
        raise ValueError(
            "TRUSTED_PROXIES takes IP addresses or CIDR ranges; "
            f"cannot parse: {', '.join(invalid)}"
        )
    return tuple(networks)


def uvicorn_forwarded_allow_ips(proxies: str) -> str | None:
    """What uvicorn may read X-Forwarded-For from, so `request.client` agrees
    with `client_address`; None leaves uvicorn on FORWARDED_ALLOW_IPS."""
    legacy = os.environ.get("FORWARDED_ALLOW_IPS")
    if not proxies:
        if legacy is not None:
            logger.warning(
                "FORWARDED_ALLOW_IPS is deprecated; set TRUSTED_PROXIES to the "
                "same list instead, which every client-IP reader uses"
            )
        return None
    if legacy is not None:
        logger.warning("FORWARDED_ALLOW_IPS is ignored because TRUSTED_PROXIES is set")
    return ",".join(str(network) for network in parse_trusted_proxies(proxies))


def forwarded_entries(lines: Iterable[str]) -> list[str]:
    """Every X-Forwarded-For line, in order, as one list: a proxy may add a line
    of its own instead of extending one."""
    return [
        value.strip() for line in lines for value in line.split(",") if value.strip()
    ]


def _is_listed(address: IPAddress, networks: tuple[IPNetwork, ...]) -> bool:
    return any(address in network for network in networks)


def _address(entry: str) -> IPAddress | None:
    """The address in a forwarded entry, without a port; IPv4-mapped IPv6 as IPv4."""
    host = entry.strip()
    if host.startswith("["):
        host = host[1:].partition("]")[0]
    elif host.count(":") == 1:
        host = host.partition(":")[0]
    try:
        if ":" in host:
            v6 = ipaddress.IPv6Address(host)
            return v6.ipv4_mapped or v6
        return ipaddress.IPv4Address(host)
    except ValueError:
        return None
