import ipaddress
import re
import socket
from typing import Callable
from urllib.parse import urljoin, urlparse, urlunparse

import idna
import requests as req

from backend.util.settings import Config

# List of IP networks to block
BLOCKED_IP_NETWORKS = [
    # --8<-- [start:BLOCKED_IP_NETWORKS]
    # IPv4 Ranges
    ipaddress.ip_network("0.0.0.0/8"),  # "This" Network
    ipaddress.ip_network("10.0.0.0/8"),  # Private-Use
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("169.254.0.0/16"),  # Link Local
    ipaddress.ip_network("172.16.0.0/12"),  # Private-Use
    ipaddress.ip_network("192.168.0.0/16"),  # Private-Use
    ipaddress.ip_network("224.0.0.0/4"),  # Multicast
    ipaddress.ip_network("240.0.0.0/4"),  # Reserved for Future Use
    # IPv6 Ranges
    ipaddress.ip_network("::1/128"),  # Loopback
    ipaddress.ip_network("fc00::/7"),  # Unique local addresses (ULA)
    ipaddress.ip_network("fe80::/10"),  # Link-local
    ipaddress.ip_network("ff00::/8"),  # Multicast
    # --8<-- [end:BLOCKED_IP_NETWORKS]
]

ALLOWED_SCHEMES = ["http", "https"]
HOSTNAME_REGEX = re.compile(r"^[A-Za-z0-9.-]+$")  # Basic DNS-safe hostname pattern


def _is_ip_blocked(ip: str) -> bool:
    """
    Checks if the IP address is in a blocked network.
    """
    ip_addr = ipaddress.ip_address(ip)
    return any(ip_addr in network for network in BLOCKED_IP_NETWORKS)


def validate_url(url: str, trusted_origins: list[str]) -> tuple[str, str]:
    """
    Validates the URL to prevent SSRF attacks by ensuring it does not point
    to a private, link-local, or otherwise blocked IP address — unless
    the hostname is explicitly trusted.

    Returns a tuple of:
    - pinned_url:  a URL that has the netloc replaced with the validated IP
    - ascii_hostname: the original ASCII hostname (IDNA-decoded) for use in the Host header
    """
    # Canonicalize URL
    url = url.strip("/ ").replace("\\", "/")
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"http://{url}"
        parsed = urlparse(url)

    # Check scheme
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Scheme '{parsed.scheme}' is not allowed. Only HTTP/HTTPS are supported."
        )

    # Validate and IDNA encode hostname
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

    # If hostname is trusted, skip IP-based checks but still return pinned URL
    if ascii_hostname in trusted_origins:
        # We still “pin” the URL but keep the hostname if it's trusted
        pinned_netloc = ascii_hostname
        if parsed.port:
            pinned_netloc += f":{parsed.port}"

        pinned_url = urlunparse(
            (
                parsed.scheme,
                pinned_netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
        return pinned_url, ascii_hostname

    # Resolve all IP addresses for the hostname
    try:
        ip_addresses = {res[4][0] for res in socket.getaddrinfo(ascii_hostname, None)}
    except socket.gaierror:
        raise ValueError(f"Unable to resolve IP address for hostname {ascii_hostname}")

    if not ip_addresses:
        raise ValueError(f"No IP addresses found for {ascii_hostname}")

    # Block any IP address that belongs to a blocked range
    for ip_str in ip_addresses:
        if _is_ip_blocked(ip_str):
            raise ValueError(
                f"Access to blocked or private IP address {ip_str} "
                f"for hostname {ascii_hostname} is not allowed."
            )

    # Pick an IP and build the pinned URL so we do not re-resolve later
    pinned_ip = next(iter(ip_addresses))  # Just pick the first valid IP
    pinned_netloc = pinned_ip
    if parsed.port:
        pinned_netloc += f":{parsed.port}"

    pinned_url = urlunparse(
        (
            parsed.scheme,
            pinned_netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )

    return pinned_url, ascii_hostname  # (pinned_url, original_hostname)


class Requests:
    """
    A wrapper around the requests library that validates URLs before
    making requests, preventing SSRF by blocking private networks and
    other disallowed address spaces.
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
        self,
        method,
        url,
        headers=None,
        allow_redirects=True,
        max_redirects=10,
        *args,
        **kwargs,
    ) -> req.Response:
        # Validate URL and get pinned URL + original hostname
        pinned_url, original_hostname = validate_url(url, self.trusted_origins)

        # Apply any extra user-defined validation/transformation
        if self.extra_url_validator is not None:
            pinned_url = self.extra_url_validator(pinned_url)

        # Merge any extra headers
        headers = dict(headers) if headers else {}
        if self.extra_headers is not None:
            headers.update(self.extra_headers)

        # Force the Host header to the original hostname
        headers["Host"] = original_hostname

        # Perform the request with redirects disabled for manual handling
        response = req.request(
            method,
            pinned_url,
            headers=headers,
            allow_redirects=False,
            *args,
            **kwargs,
        )

        if self.raise_for_status:
            response.raise_for_status()

        # If allowed and a redirect is received, follow the redirect manually
        if allow_redirects and response.is_redirect:
            if max_redirects <= 0:
                raise Exception("Too many redirects.")

            location = response.headers.get("Location")
            if not location:
                return response

            # The base URL is the pinned_url we just used
            # so that relative redirects resolve correctly.
            new_raw_url = urljoin(pinned_url, location)

            # Validate/pin the redirect URL
            new_pinned_url, new_original_hostname = validate_url(
                new_raw_url, self.trusted_origins
            )
            if self.extra_url_validator is not None:
                new_pinned_url = self.extra_url_validator(new_pinned_url)

            # Carry forward the same headers but update Host
            new_headers = dict(headers)
            new_headers["Host"] = new_original_hostname

            return self.request(
                method,
                new_pinned_url,
                headers=new_headers,
                allow_redirects=allow_redirects,
                max_redirects=max_redirects - 1,
                *args,
                **kwargs,
            )

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
