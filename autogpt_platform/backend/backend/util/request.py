import ipaddress
import socket
from typing import Callable
from urllib.parse import urlparse

import requests as req

from backend.util.settings import Config

# List of IP networks to block
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),  # "This" Network
    ipaddress.ip_network("10.0.0.0/8"),  # Private-Use
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("169.254.0.0/16"),  # Link Local
    ipaddress.ip_network("172.16.0.0/12"),  # Private-Use
    ipaddress.ip_network("192.168.0.0/16"),  # Private-Use
    ipaddress.ip_network("224.0.0.0/4"),  # Multicast
    ipaddress.ip_network("240.0.0.0/4"),  # Reserved for Future Use
]


def is_ip_blocked(ip: str) -> bool:
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
    url = url.strip().strip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed_url = urlparse(url)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError(f"Invalid URL: Unable to determine hostname from {url}")

    if any(hostname == origin for origin in trusted_origins):
        return url

    # Resolve all IP addresses for the hostname
    ip_addresses = {result[4][0] for result in socket.getaddrinfo(hostname, None)}
    if not ip_addresses:
        raise ValueError(f"Unable to resolve IP address for {hostname}")

    # Check if all IP addresses are global
    for ip in ip_addresses:
        if is_ip_blocked(ip):
            raise ValueError(
                f"Access to private IP address at {hostname}: {ip} is not allowed."
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
