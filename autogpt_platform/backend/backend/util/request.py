import asyncio
import ipaddress
import re
import socket
import ssl
from io import BytesIO
from typing import Any, Callable, Optional
from urllib.parse import ParseResult as URL
from urllib.parse import quote, urljoin, urlparse

import aiohttp
import idna
from aiohttp import FormData, abc
from tenacity import (
    RetryCallState,
    retry,
    retry_if_result,
    stop_after_attempt,
    wait_exponential_jitter,
)

from backend.util.json import loads


class HTTPClientError(Exception):
    """4xx client errors (400-499)"""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class HTTPServerError(Exception):
    """5xx server errors (500-599)"""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


# Default User-Agent for all requests
DEFAULT_USER_AGENT = "AutoGPT-Platform/1.0 (https://github.com/Significant-Gravitas/AutoGPT; info@agpt.co) aiohttp"

# Retry status codes for which we will automatically retry the request
THROTTLE_RETRY_STATUS_CODES: set[int] = {429, 500, 502, 503, 504, 408}

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


def _remove_insecure_headers(headers: dict, old_url: URL, new_url: URL) -> dict:
    """
    Removes sensitive headers (Authorization, Proxy-Authorization, Cookie)
    if the scheme/host/port of new_url differ from old_url.
    """
    if (
        (old_url.scheme != new_url.scheme)
        or (old_url.hostname != new_url.hostname)
        or (old_url.port != new_url.port)
    ):
        headers.pop("Authorization", None)
        headers.pop("Proxy-Authorization", None)
        headers.pop("Cookie", None)
    return headers


class HostResolver(abc.AbstractResolver):
    """
    A custom resolver that connects to specified IP addresses but still
    sets the TLS SNI to the original host name so the cert can match.
    """

    def __init__(self, ssl_hostname: str, ip_addresses: list[str]):
        self.ssl_hostname = ssl_hostname
        self.ip_addresses = ip_addresses
        self._default = aiohttp.AsyncResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        if host == self.ssl_hostname:
            results = []
            for ip in self.ip_addresses:
                results.append(
                    {
                        "hostname": self.ssl_hostname,
                        "host": ip,
                        "port": port,
                        "family": family,
                        "proto": 0,
                        "flags": socket.AI_NUMERICHOST,
                    }
                )
            return results
        return await self._default.resolve(host, port, family)

    async def close(self):
        await self._default.close()


async def _resolve_host(hostname: str) -> list[str]:
    """
    Resolves the hostname to a list of IP addresses (IPv4 first, then IPv6).
    """
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise ValueError(f"Unable to resolve IP address for hostname {hostname}")

    ip_list = [info[4][0] for info in infos]
    ipv4 = [ip for ip in ip_list if ":" not in ip]
    ipv6 = [ip for ip in ip_list if ":" in ip]
    ip_addresses = ipv4 + ipv6

    if not ip_addresses:
        raise ValueError(f"No IP addresses found for {hostname}")
    return ip_addresses


async def validate_url(
    url: str, trusted_origins: list[str]
) -> tuple[URL, bool, list[str]]:
    """
    Validates the URL to prevent SSRF attacks by ensuring it does not point
    to a private, link-local, or otherwise blocked IP address — unless
    the hostname is explicitly trusted.

    Returns:
        str: The validated, canonicalized, parsed URL
        is_trusted: Boolean indicating if the hostname is in trusted_origins
        ip_addresses: List of IP addresses for the host; empty if the host is trusted
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

    # Check if hostname is trusted
    is_trusted = ascii_hostname in trusted_origins

    # If not trusted, validate IP addresses
    ip_addresses: list[str] = []
    if not is_trusted:
        # Resolve all IP addresses for the hostname
        ip_addresses = await _resolve_host(ascii_hostname)

        # Block any IP address that belongs to a blocked range
        for ip_str in ip_addresses:
            if _is_ip_blocked(ip_str):
                raise ValueError(
                    f"Access to blocked or private IP address {ip_str} "
                    f"for hostname {ascii_hostname} is not allowed."
                )

    # Reconstruct the netloc with IDNA-encoded hostname and preserve port
    netloc = ascii_hostname
    if parsed.port:
        netloc = f"{ascii_hostname}:{parsed.port}"

    return (
        URL(
            parsed.scheme,
            netloc,
            quote(parsed.path, safe="/%:@"),
            parsed.params,
            parsed.query,
            parsed.fragment,
        ),
        is_trusted,
        ip_addresses,
    )


def pin_url(url: URL, ip_addresses: Optional[list[str]] = None) -> URL:
    """
    Pins a URL to a specific IP address to prevent DNS rebinding attacks.

    Args:
        url: The original URL
        ip_addresses: List of IP addresses corresponding to the URL's host

    Returns:
        pinned_url: The URL with hostname replaced with IP address
    """
    if not url.hostname:
        raise ValueError(f"URL has no hostname: {url}")

    if not ip_addresses:
        # Resolve all IP addresses for the hostname
        # (This call is blocking; ensure to call async _resolve_host before if possible)
        ip_addresses = []
        # You may choose to raise or call synchronous resolve here; for simplicity, leave empty.

    # Pin to the first valid IP (for SSRF defense)
    pinned_ip = ip_addresses[0]

    # If it's IPv6, bracket it
    if ":" in pinned_ip:
        pinned_netloc = f"[{pinned_ip}]"
    else:
        pinned_netloc = pinned_ip

    if url.port:
        pinned_netloc += f":{url.port}"

    return URL(
        url.scheme,
        pinned_netloc,
        url.path,
        url.params,
        url.query,
        url.fragment,
    )


ClientResponse = aiohttp.ClientResponse
ClientResponseError = aiohttp.ClientResponseError


class Response:
    """
    Buffered wrapper around aiohttp.ClientResponse that does *not* require
    callers to manage connection or session lifetimes.
    """

    def __init__(
        self,
        *,
        response: ClientResponse,
        url: str,
        body: bytes,
    ):
        self.status: int = response.status
        self.headers = response.headers
        self.reason: str | None = response.reason
        self.request_info = response.request_info
        self.url: str = url
        self.content: bytes = body  # raw bytes

    def json(self, encoding: str | None = None, **kwargs) -> dict:
        """
        Parse the body as JSON and return the resulting Python object.
        """
        return loads(
            self.content.decode(encoding or "utf-8", errors="replace"), **kwargs
        )

    def text(self, encoding: str | None = None) -> str:
        """
        Decode the body to a string.  Encoding is guessed from the
        Content-Type header if not supplied.
        """
        if encoding is None:
            # Try to extract charset from headers; fall back to UTF-8
            ctype = self.headers.get("content-type", "")
            match = re.search(r"charset=([^\s;]+)", ctype, flags=re.I)
            encoding = match.group(1) if match else None
        return self.content.decode(encoding or "utf-8", errors="replace")

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


def _return_last_result(retry_state: RetryCallState) -> "Response":
    """
    Ensure the final attempt's response is returned when retrying stops.
    """
    if retry_state.outcome is None:
        raise RuntimeError("Retry state is missing an outcome.")

    exception = retry_state.outcome.exception()
    if exception is not None:
        raise exception

    return retry_state.outcome.result()


class Requests:
    """
    A wrapper around an aiohttp ClientSession that validates URLs before
    making requests, preventing SSRF by blocking private networks and
    other disallowed address spaces.
    """

    def __init__(
        self,
        trusted_origins: list[str] | None = None,
        raise_for_status: bool = True,
        extra_url_validator: Callable[[URL], URL] | None = None,
        extra_headers: dict[str, str] | None = None,
        retry_max_wait: float = 300.0,
        retry_max_attempts: int | None = None,
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
        self.retry_max_wait = retry_max_wait
        if retry_max_attempts is not None and retry_max_attempts < 1:
            raise ValueError("retry_max_attempts must be None or >= 1")
        self.retry_max_attempts = retry_max_attempts

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict] = None,
        files: list[tuple[str, tuple[str, BytesIO, str]]] | None = None,
        data: Any | None = None,
        json: Any | None = None,
        allow_redirects: bool = True,
        max_redirects: int = 10,
        **kwargs,
    ) -> Response:
        retry_kwargs: dict[str, Any] = {
            "wait": wait_exponential_jitter(max=self.retry_max_wait),
            "retry": retry_if_result(lambda r: r.status in THROTTLE_RETRY_STATUS_CODES),
            "reraise": True,
        }

        if self.retry_max_attempts is not None:
            retry_kwargs["stop"] = stop_after_attempt(self.retry_max_attempts)
            retry_kwargs["retry_error_callback"] = _return_last_result

        @retry(**retry_kwargs)
        async def _make_request() -> Response:
            return await self._request(
                method=method,
                url=url,
                headers=headers,
                files=files,
                data=data,
                json=json,
                allow_redirects=allow_redirects,
                max_redirects=max_redirects,
                **kwargs,
            )

        return await _make_request()

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict] = None,
        files: list[tuple[str, tuple[str, BytesIO, str]]] | None = None,
        data: Any | None = None,
        json: Any | None = None,
        allow_redirects: bool = True,
        max_redirects: int = 10,
        **kwargs,
    ) -> Response:
        # Convert auth tuple to aiohttp.BasicAuth if necessary
        if "auth" in kwargs and isinstance(kwargs["auth"], tuple):
            kwargs["auth"] = aiohttp.BasicAuth(*kwargs["auth"])

        if files is not None:
            if json is not None:
                raise ValueError(
                    "Cannot mix file uploads with JSON body; "
                    "use 'data' for extra form fields instead."
                )

            form = FormData(quote_fields=False)
            # add normal form fields first
            if isinstance(data, dict):
                for k, v in data.items():
                    form.add_field(k, str(v))
            elif data is not None:
                raise ValueError(
                    "When uploading files, 'data' must be a dict of form fields."
                )

            # add the file parts
            for field_name, (filename, fh, content_type) in files:
                form.add_field(
                    name=field_name,
                    value=fh,
                    filename=filename,
                    content_type=content_type or "application/octet-stream",
                )

            data = form

        # Validate URL and get trust status
        parsed_url, is_trusted, ip_addresses = await validate_url(
            url, self.trusted_origins
        )

        # Apply any extra user-defined validation/transformation
        if self.extra_url_validator is not None:
            parsed_url = self.extra_url_validator(parsed_url)

        # Pin the URL if untrusted
        hostname = parsed_url.hostname
        if hostname is None:
            raise ValueError(f"Invalid URL: Unable to determine hostname of {url}")

        original_url = parsed_url.geturl()
        connector: Optional[aiohttp.TCPConnector] = None
        if not is_trusted:
            # Replace hostname with IP for connection but preserve SNI via resolver
            resolver = HostResolver(ssl_hostname=hostname, ip_addresses=ip_addresses)
            ssl_context = ssl.create_default_context()
            connector = aiohttp.TCPConnector(resolver=resolver, ssl=ssl_context)
        session_kwargs = {}
        if connector:
            session_kwargs["connector"] = connector

        # Merge any extra headers
        req_headers = dict(headers) if headers else {}
        if self.extra_headers is not None:
            req_headers.update(self.extra_headers)

        # Set default User-Agent if not provided
        if "User-Agent" not in req_headers and "user-agent" not in req_headers:
            req_headers["User-Agent"] = DEFAULT_USER_AGENT

        # Override Host header if using IP connection
        if connector:
            req_headers["Host"] = hostname

        # Override data if files are provided
        # Set max_field_size to handle servers with large headers (e.g., long CSP headers)
        # Default is 8190 bytes, we increase to 16KB to accommodate legitimate large headers
        session_kwargs["max_field_size"] = 16384

        async with aiohttp.ClientSession(**session_kwargs) as session:
            # Perform the request with redirects disabled for manual handling
            async with session.request(
                method,
                parsed_url.geturl(),
                headers=req_headers,
                allow_redirects=False,
                data=data,
                json=json,
                **kwargs,
            ) as response:

                if self.raise_for_status:
                    try:
                        response.raise_for_status()
                    except ClientResponseError as e:
                        body = await response.read()
                        error_message = f"HTTP {response.status} Error: {response.reason}, Body: {body.decode(errors='replace')}"

                        # Raise specific exceptions based on status code range
                        if 400 <= response.status <= 499:
                            raise HTTPClientError(error_message, response.status) from e
                        elif 500 <= response.status <= 599:
                            raise HTTPServerError(error_message, response.status) from e
                        else:
                            # Generic fallback for other HTTP errors
                            raise Exception(error_message) from e

                # If allowed and a redirect is received, follow the redirect manually
                if allow_redirects and response.status in (301, 302, 303, 307, 308):
                    if max_redirects <= 0:
                        raise Exception("Too many redirects.")

                    location = response.headers.get("Location")
                    if not location:
                        return Response(
                            response=response,
                            url=original_url,
                            body=await response.read(),
                        )

                    # The base URL is the pinned_url we just used
                    # so that relative redirects resolve correctly.
                    redirect_url = urlparse(urljoin(parsed_url.geturl(), location))
                    # Carry forward the same headers but update Host
                    new_headers = _remove_insecure_headers(
                        req_headers, parsed_url, redirect_url
                    )

                    return await self.request(
                        method,
                        redirect_url.geturl(),
                        headers=new_headers,
                        allow_redirects=allow_redirects,
                        max_redirects=max_redirects - 1,
                        files=files,
                        data=data,
                        json=json,
                        **kwargs,
                    )

                # Reset response URL to original host for clarity
                if parsed_url.hostname != hostname:
                    try:
                        response.url = original_url  # type: ignore
                    except Exception:
                        pass

                return Response(
                    response=response,
                    url=original_url,
                    body=await response.read(),
                )

    async def get(self, url: str, *args, **kwargs) -> Response:
        return await self.request("GET", url, *args, **kwargs)

    async def post(self, url: str, *args, **kwargs) -> Response:
        return await self.request("POST", url, *args, **kwargs)

    async def put(self, url: str, *args, **kwargs) -> Response:
        return await self.request("PUT", url, *args, **kwargs)

    async def delete(self, url: str, *args, **kwargs) -> Response:
        return await self.request("DELETE", url, *args, **kwargs)

    async def head(self, url: str, *args, **kwargs) -> Response:
        return await self.request("HEAD", url, *args, **kwargs)

    async def options(self, url: str, *args, **kwargs) -> Response:
        return await self.request("OPTIONS", url, *args, **kwargs)

    async def patch(self, url: str, *args, **kwargs) -> Response:
        return await self.request("PATCH", url, *args, **kwargs)
