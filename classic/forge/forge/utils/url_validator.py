import functools
import ipaddress
import re
import socket
from inspect import signature
from typing import Callable, ParamSpec, TypeVar
from urllib.parse import urljoin, urlparse

P = ParamSpec("P")
T = TypeVar("T")


def validate_url(func: Callable[P, T]) -> Callable[P, T]:
    """
    The method decorator validate_url is used to validate urls for any command that
    requires a url as an argument.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        url = bound_args.arguments.get("url")
        if url is None:
            raise ValueError("URL is required for this function")

        if not re.match(r"^https?://", url):
            raise ValueError(
                "Invalid URL format: URL must start with http:// or https://"
            )
        if not is_valid_url(url):
            raise ValueError("Missing Scheme or Network location")
        if check_local_file_access(url):
            raise ValueError("Access to local files is restricted")
        if len(url) > 2000:
            raise ValueError("URL is too long")
        if not check_public_address(url):
            raise ValueError("Access to internal/private addresses is restricted")

        bound_args.arguments["url"] = sanitize_url(url)

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper  # type: ignore


def is_valid_url(url: str) -> bool:
    """Check if the URL is valid

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def sanitize_url(url: str) -> str:
    """Sanitize the URL

    Args:
        url (str): The URL to sanitize

    Returns:
        str: The sanitized URL
    """
    parsed_url = urlparse(url)
    reconstructed_url = f"{parsed_url.path}{parsed_url.params}?{parsed_url.query}"
    return urljoin(url, reconstructed_url)


def check_local_file_access(url: str) -> bool:
    """Check if the URL is a local file

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is a local file, False otherwise
    """
    # List of local file prefixes
    local_file_prefixes = [
        "file:///",
        "file://localhost",
    ]

    return any(url.startswith(prefix) for prefix in local_file_prefixes)


# Cloud instance metadata endpoint commonly abused via SSRF.
METADATA_IP = "169.254.169.254"


def check_public_address(url: str) -> bool:
    """Check that the URL's hostname resolves only to public addresses (SSRF guard).

    Resolves the hostname and rejects the URL if any resolved IP is private,
    loopback, link-local, reserved, multicast, unspecified, or the cloud metadata
    IP. ``ipaddress.ip_address`` canonicalizes numeric/alternate encodings (decimal,
    hex, IPv4-mapped IPv6), so those are caught too.

    Note: this is resolve-time validation. DNS rebinding (TOCTOU between this check
    and the actual request) is a known residual limitation not addressed here.

    Args:
        url (str): The URL to check

    Returns:
        bool: True if every resolved address is public, False otherwise
    """
    hostname = urlparse(url).hostname
    if not hostname:
        return False

    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        # Hostname did not resolve; nothing private to reach, leave it to the caller.
        return True

    for addr_info in addr_infos:
        ip = ipaddress.ip_address(addr_info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
            or str(ip) == METADATA_IP
        ):
            return False

    return True
