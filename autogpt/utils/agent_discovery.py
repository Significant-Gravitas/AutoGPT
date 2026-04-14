"""Agent Discovery Protocol (ADP) v0.1 -- discover agent services at any domain.

The Agent Discovery Protocol defines a standard for domains to publish available
AI agent services at /.well-known/agent-discovery.json. This module provides a
lightweight client for agents to discover services before interacting with a
domain.

Spec: https://github.com/walkojas-boop/agent-discovery-protocol

Usage:
    from autogpt.utils.agent_discovery import discover_services

    services = discover_services("walkosystems.com")
    if services:
        memory = services.get_service("memory")
        if memory:
            print(f"Memory endpoint: {memory['endpoint']}")
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, dict | None]] = {}
_CACHE_TTL = 3600  # 1 hour

# FQDN validation: letters, digits, hyphens, dots. No IP literals,
# no embedded schemes, no ports, no userinfo.
_FQDN_RE = re.compile(
    r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,}$"
)

# Private/reserved networks to reject (SSRF protection)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _validate_domain(domain: str) -> str | None:
    """Validate domain is a safe FQDN. Returns error message or None."""
    if not domain or not isinstance(domain, str):
        return "domain must be a non-empty string"
    if not _FQDN_RE.match(domain):
        return f"invalid domain format: {domain!r}"
    # Resolve and check for private IPs
    import socket

    try:
        addrs = socket.getaddrinfo(domain, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return f"domain does not resolve: {domain}"
    for family, _, _, _, sockaddr in addrs:
        ip = ipaddress.ip_address(sockaddr[0])
        for net in _BLOCKED_NETWORKS:
            if ip in net:
                return f"domain resolves to blocked address: {ip}"
    return None


class DiscoveryResult:
    """Parsed agent discovery document."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self._services = {
            s["name"]: s for s in data.get("services", [])
        }

    @property
    def domain(self) -> str:
        return self._data.get("domain", "")

    @property
    def version(self) -> str:
        return self._data.get("agent_discovery_version", "")

    @property
    def services(self) -> dict[str, dict]:
        return self._services

    @property
    def trust(self) -> dict:
        return self._data.get("trust", {})

    def get_service(self, name: str) -> dict | None:
        """Get a service by name."""
        return self._services.get(name)

    def list_services(self) -> list[str]:
        """List all available service names."""
        return list(self._services.keys())

    def has_service(self, name: str) -> bool:
        """Check if a service is available."""
        return name in self._services

    def __repr__(self) -> str:
        return (
            f"DiscoveryResult(domain={self.domain!r}, "
            f"services={self.list_services()})"
        )


def discover_services(
    domain: str,
    *,
    timeout: float = 5.0,
    use_cache: bool = True,
) -> DiscoveryResult | None:
    """Discover agent services at a domain via ADP.

    Fetches /.well-known/agent-discovery.json from the given domain
    and returns a parsed result. Returns None if the domain doesn't
    implement ADP.

    Args:
        domain: FQDN to check (e.g., "walkosystems.com").
            IP literals, private ranges, and non-FQDN inputs
            are rejected.
        timeout: Request timeout in seconds.
        use_cache: Whether to cache results (default: True,
            1-hour TTL).

    Returns:
        DiscoveryResult if the domain publishes agent services,
        None otherwise.

    Raises:
        ValueError: If domain fails SSRF validation.
    """
    # SSRF protection: validate domain before any network I/O
    validation_error = _validate_domain(domain)
    if validation_error:
        raise ValueError(validation_error)

    if use_cache and domain in _cache:
        cached_at, cached_result = _cache[domain]
        if time.time() - cached_at < _CACHE_TTL:
            return (
                DiscoveryResult(cached_result)
                if cached_result
                else None
            )

    url = f"https://{domain}/.well-known/agent-discovery.json"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "agent-discovery/0.1"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read())
                if use_cache:
                    _cache[domain] = (time.time(), data)
                return DiscoveryResult(data)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ) as e:
        logger.debug("ADP: no discovery at %s (%s)", domain, e)

    if use_cache:
        _cache[domain] = (time.time(), None)
    return None
