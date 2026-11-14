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

IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


# Documented in docs/platform/client-ip.md: update it with any change here.
def client_address(peer: str, forwarded_for: Iterable[str], proxies: str) -> str:
    """The first X-Forwarded-For entry from the right that `proxies` does not list,
    read only from a listed peer: what uvicorn makes `request.client`.

    As in uvicorn, an entry that is not an address counts as unlisted, an empty
    one leaves the peer, and a header of listed entries only names its leftmost.
    """
    networks = parse_trusted_proxies(proxies)
    lines = list(forwarded_for)
    if not lines or not _is_listed(peer, networks):
        return peer
    entries = [entry.strip() for entry in ", ".join(lines).split(",")]
    for entry in reversed(entries):
        host = _host(entry)
        if not _is_listed(host, networks):
            return host or peer
    return _host(entries[0]) or peer


@functools.cache
def parse_trusted_proxies(value: str) -> tuple[IPNetwork, ...]:
    """`value`'s comma-separated addresses and CIDR ranges, refusing any other entry.

    single-container/runtime_config.py's validate_trusted_proxies keeps the same
    contract and message by hand, since the entrypoint cannot import the backend;
    both are tested against trusted_proxies_vectors.json."""
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


def _is_listed(host: str, networks: tuple[IPNetwork, ...]) -> bool:
    # No IPv4-mapped unwrap: uvicorn matches ::ffff:a.b.c.d only against IPv6 ranges.
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in network for network in networks)


def _host(entry: str) -> str:
    """The host in a forwarded entry, as uvicorn's _parse_host_port reads it:
    only a port may follow the address, or the entry stays whole."""
    if entry.startswith("["):
        host, bracket, rest = entry[1:].partition("]")
        return host if bracket and (not rest or rest.startswith(":")) else entry
    if entry.count(":") == 1:
        host, _, port = entry.partition(":")
        try:
            int(port)
        except ValueError:
            return entry
        return host
    return entry
