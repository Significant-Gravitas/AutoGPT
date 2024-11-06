import ipaddress
import socket
from typing import Callable
from urllib.parse import urlparse

import requests as req

from backend.util.settings import Config


def is_ip_allowed(ip: str) -> bool:
    """
    Checks if the IP address is allowed (i.e., it's a global IP address).
    """
    ip_addr = ipaddress.ip_address(ip)
    return ip_addr.is_global


def validate_url(url: str, trusted_origins: list[str]) -> str:
    """
    Validates the URL to prevent SSRF attacks by ensuring it does not point to a private
    or untrusted IP address, unless whitelisted.
    """
    if any(url.startswith(origin) for origin in trusted_origins):
        return url

    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(f"Invalid URL: Unable to determine hostname from {url}")

    try:
        # Resolve all IP addresses for the hostname
        ip_addresses = {result[4][0] for result in socket.getaddrinfo(hostname, None)}
        # Check if all IP addresses are global
        if all(is_ip_allowed(ip) for ip in ip_addresses):
            return url
        else:
            raise ValueError(
                f"Access to private or untrusted IP address at {hostname} is not allowed."
            )
    except Exception as e:
        raise ValueError(f"Invalid or unresolvable URL: {url}") from e


class Requests:
    """
    A wrapper around the requests library that validates URLs before making requests.
    """

    def __init__(
        self,
        trusted_origins: list[str],
        raise_for_status: bool = True,
        extra_url_validator: Callable[[str], str] | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        self.trusted_origins = trusted_origins
        self.raise_for_status = raise_for_status
        self.extra_url_validator = extra_url_validator
        self.extra_headers = extra_headers

    def request(self, method, url, headers=None, *args, **kwargs) -> req.Response:
        if self.extra_headers is not None:
            headers = {**(headers or {}), **self.extra_headers}

        if self.extra_url_validator is not None:
            url = self.extra_url_validator(url)
        url = validate_url(url, self.trusted_origins)

        response = req.request(method, url, headers=headers, *args, **kwargs)
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


requests = Requests(Config().trust_endpoints_for_requests)
