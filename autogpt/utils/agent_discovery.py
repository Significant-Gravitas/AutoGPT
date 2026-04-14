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
import socket
import time
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, dict | None]] = {}
_CACHE_TTL = 3600  # 1 hour

# FQDN validation: letters, digits, hyphens, dots. No IP literals,
# no embedded schemes, no ports, no userinfo.
_FQDN_RE = re.compile(
    r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)"
    r"(\.[A-Za-z0-9-]{1,63})*"
    r"\.[A-Za-z]{2,}$"
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


def _is_blocked_ip(ip_str: str) -> bool:
    """Check if an IP address is in a blocked/private range."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return any(ip in net for net in _BLOCKED_NETWORKS)


def _validate_domain(domain: str) -> str | None:
    """Validate domain is a safe FQDN.

    Returns error message string on failure, None on success.
    """
    if not domain or not isinstance(domain, str):
        return "domain must be a non-empty string"
    if not _FQDN_RE.match(domain):
        return f"invalid domain format: {domain!r}"
    return None


def _resolve_and_validate(domain: str) -> tuple[str, str] | None:
    """Resolve domain and validate IPs are not private.

    Returns (ip, family_str) on success, raises ValueError on
    failure. Uses the first non-blocked address found.
    """
    try:
        addrs = socket.getaddrinfo(
            domain, 443, proto=socket.IPPROTO_TCP
        )
    except socket.gaierror:
        raise ValueError(
            f"domain does not resolve: {domain}"
        )

    for _family, _, _, _, sockaddr in addrs:
        ip_str = sockaddr[0]
        if _is_blocked_ip(ip_str):
            raise ValueError(
                f"domain resolves to blocked address: {ip_str}"
            )

    # Return first address for pinned connection
    first_ip = addrs[0][4][0]
    return first_ip


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Disable automatic redirect following to prevent SSRF bypass."""

    def redirect_request(
        self, req, fp, code, msg, headers, newurl
    ):
        raise urllib.error.HTTPError(
            req.full_url,
            code,
            f"Redirect to {newurl} blocked (SSRF protection)",
            headers,
            fp,
        )


class DiscoveryResult:
    """Parsed agent discovery document."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = deepcopy(data)
        self._services = {
            s["name"]: s
            for s in self._data.get("services", [])
        }

    @property
    def domain(self) -> str:
        return self._data.get("domain", "")

    @property
    def version(self) -> str:
        return self._data.get("agent_discovery_version", "")

    @property
    def services(self) -> dict[str, dict]:
        return deepcopy(self._services)

    @property
    def trust(self) -> dict:
        return deepcopy(self._data.get("trust", {}))

    def get_service(self, name: str) -> dict | None:
        """Get a service by name."""
        svc = self._services.get(name)
        return deepcopy(svc) if svc else None

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
    # SSRF step 1: validate domain format
    fmt_error = _validate_domain(domain)
    if fmt_error:
        raise ValueError(fmt_error)

    # SSRF step 2: resolve DNS and reject private IPs
    pinned_ip = _resolve_and_validate(domain)

    # Check cache after validation
    if use_cache and domain in _cache:
        cached_at, cached_result = _cache[domain]
        if time.time() - cached_at < _CACHE_TTL:
            return (
                DiscoveryResult(cached_result)
                if cached_result
                else None
            )

    # SSRF step 3: connect to the pinned IP directly
    # with Host header set to original domain.
    # This prevents DNS rebinding (TOCTOU) attacks.
    url = f"https://{pinned_ip}/.well-known/agent-discovery.json"

    # Build opener with no redirect following
    opener = urllib.request.build_opener(_NoRedirectHandler)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "agent-discovery/0.1",
            "Host": domain,
        },
    )
    try:
        with opener.open(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                data = json.loads(resp.read())
                if use_cache:
                    _cache[domain] = (time.time(), data)
                return DiscoveryResult(data)
    except urllib.error.HTTPError as e:
        logger.debug(
            "ADP: HTTP %d at %s (%s)", e.code, domain, e
        )
        # Only negative-cache authoritative misses
        if use_cache and e.code in {404, 410}:
            _cache[domain] = (time.time(), None)
        return None
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as e:
        # Transient failures: do NOT negative-cache
        logger.debug(
            "ADP: transient failure at %s (%s)", domain, e
        )
        return None

    # Non-2xx response that isn't an HTTPError
    if use_cache:
        _cache[domain] = (time.time(), None)
    return None
