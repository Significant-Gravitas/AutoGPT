import ipaddress
import re
import socket
from typing import Callable
from urllib.parse import urlparse, urlunparse

import idna
import requests as req

from backend.util.settings import Config

# List of IP networks to block
BLOCKED_IP_NETWORKS = [
    # --8<-- [start:BLOCKED_IP_NETWORKS]
    ipaddress.ip_network("0.0.0.0/8"),  # "This" Network
    ipaddress.ip_network("10.0.0.0/8"),  # Private-Use
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("169.254.0.0/16"),  # Link Local
    ipaddress.ip_network("172.16.0.0/12"),  # Private-Use
    ipaddress.ip_network("192.168.0.0/16"),  # Private-Use
    ipaddress.ip_network("224.0.0.0/4"),  # Multicast
    ipaddress.ip_network("240.0.0.0/4"),  # Reserved for Future Use
    # --8<-- [end:BLOCKED_IP_NETWORKS]
]

ALLOWED_SCHEMES = ["http", "https"]
HOSTNAME_REGEX = re.compile(r"^[A-Za-z0-9.-]+$")  # Basic DNS-safe hostname pattern


def _canonicalize_url(url: str) -> str:
    # Strip spaces and trailing slashes
    url = url.strip().strip("/")
    # Ensure the URL starts with http:// or https://
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    # Replace backslashes with forward slashes to avoid parsing ambiguities
    url = url.replace("\\", "/")
    return url


def _is_ip_blocked(ip: str) -> bool:
    """
    Checks if the IP address is in a blocked network.
    """
    ip_addr = ipaddress.ip_address(ip)
    return any(ip_addr in network for network in BLOCKED_IP_NETWORKS)


def validate_url(url: str, trusted_origins: list[str]) -> str:
    """
    Validates the URL to prevent SSRF attacks by ensuring it does not point to a private
    or untrusted IP address, unless whitelisted.
    """
    url = _canonicalize_url(url)
    parsed = urlparse(url)

    # Check scheme
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Scheme '{parsed.scheme}' is not allowed. Only HTTP/HTTPS are supported."
        )

    # Validate and IDNA encode the hostname
    if not parsed.hostname:
        raise ValueError("Invalid URL: No hostname found.")

    # IDNA encode to prevent Unicode domain attacks
    try:
        ascii_hostname = idna.encode(parsed.hostname).decode("ascii")
    except idna.IDNAError:
        raise ValueError("Invalid hostname with unsupported characters.")

    # Check hostname characters
    if not HOSTNAME_REGEX.match(ascii_hostname):
        raise ValueError("Hostname contains invalid characters.")

    # Rebuild the URL with the normalized, IDNA-encoded hostname
    parsed = parsed._replace(netloc=ascii_hostname)
    url = str(urlunparse(parsed))

    # Check if hostname is a trusted origin (exact match)
    if ascii_hostname in trusted_origins:
        return url

    # Resolve all IP addresses for the hostname
    try:
        ip_addresses = {res[4][0] for res in socket.getaddrinfo(ascii_hostname, None)}
    except socket.gaierror:
        raise ValueError(f"Unable to resolve IP address for hostname {ascii_hostname}")

    if not ip_addresses:
        raise ValueError(f"No IP addresses found for {ascii_hostname}")

    # Check if any resolved IP address falls into blocked ranges
    for ip in ip_addresses:
        if _is_ip_blocked(ip):
            raise ValueError(
                f"Access to private IP address {ip} for hostname {ascii_hostname} is not allowed."
            )

    return url


class Requests:
    """
    A wrapper around the requests library that validates URLs before making requests.
    """

    def __init__(
        self,
        trusted_origins: list[str] | None = None,
        raise_for_status: bool = True,
        extra_url_validator: Callable[[str], str] | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        self.trusted_origins = []
        for url in trusted_origins or []:
            hostname = urlparse(url).hostname
            if not hostname:
                raise ValueError(f"Invalid URL: Unable to determine hostname of {url}")
            self.trusted_origins.append(hostname)

        self.raise_for_status = raise_for_status
        self.extra_url_validator = extra_url_validator
        self.extra_headers = extra_headers

    def request(
        self, method, url, headers=None, allow_redirects=False, *args, **kwargs
    ) -> req.Response:
        if self.extra_headers is not None:
            headers = {**(headers or {}), **self.extra_headers}

        url = validate_url(url, self.trusted_origins)
        if self.extra_url_validator is not None:
            url = self.extra_url_validator(url)

        response = req.request(
            method,
            url,
            headers=headers,
            allow_redirects=allow_redirects,
            *args,
            **kwargs,
        )
        if self.raise_for_status:
            response.raise_for_status()

        return response

    def get(self, url, *args, **kwargs) -> req.Response:
        return self.request("GET", url, *args, **kwargs)

    def post(self, url, *args, **kwargs) -> req.Response:
        return self.request("POST", url, *args, **kwargs)

    def put(self, url, *args, **kwargs) -> req.Response:
        return self.request("PUT", url, *args, **kwargs)

    def delete(self, url, *args, **kwargs) -> req.Response:
        return self.request("DELETE", url, *args, **kwargs)

    def head(self, url, *args, **kwargs) -> req.Response:
        return self.request("HEAD", url, *args, **kwargs)

    def options(self, url, *args, **kwargs) -> req.Response:
        return self.request("OPTIONS", url, *args, **kwargs)

    def patch(self, url, *args, **kwargs) -> req.Response:
        return self.request("PATCH", url, *args, **kwargs)


requests = Requests(trusted_origins=Config().trust_endpoints_for_requests)
