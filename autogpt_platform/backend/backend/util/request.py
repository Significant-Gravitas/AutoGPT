import asyncio
import codecs
import ipaddress
import logging
import random
import re
import socket
import ssl
from asyncio import sleep
from collections.abc import Iterable
from typing import Any, Callable, Optional
from urllib.parse import ParseResult as URL
from urllib.parse import quote, urljoin, urlparse

import aiohttp
import idna
from aiohttp import abc
from aiohttp.client import DEFAULT_TIMEOUT
from pydantic import BaseModel, ConfigDict

from backend.util.json import loads
from backend.util.request_body import PreparedBody, Uploads, prepare_body
from backend.util.settings import Config
from backend.util.url_redaction import is_credential_header

logger = logging.getLogger(__name__)


class HTTPClientError(Exception):
    """4xx client errors (400-499)"""

    def __init__(self, message: str, status_code: int, body: bytes | None = None):
        super().__init__(message)
        self.status_code = status_code
        # The raw response body, so a caller can read a structured error code
        # (e.g. OAuth's ``{"error": "invalid_grant"}``) without parsing the message.
        self.body = body


class HTTPServerError(Exception):
    """5xx server errors (500-599)"""

    def __init__(self, message: str, status_code: int, body: bytes | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


# Default User-Agent for all requests
class ResponseTooLargeError(ValueError):
    """The response body is over the caller's size limit."""

    def __init__(self, limit: int):
        super().__init__(f"Response body is larger than {limit} bytes")
        self.limit = limit


DEFAULT_USER_AGENT = "AutoGPT-Platform/1.0 (https://github.com/Significant-Gravitas/AutoGPT; info@agpt.co) aiohttp"

# A response is read in chunks and dropped once it passes the limit, so a
# large or endless body can't fill the process's memory before anyone checks
# its size. Downloads share the file-size setting.
DEFAULT_MAX_RESPONSE_BYTES = Config().max_file_size_mb * 1024 * 1024
RESPONSE_READ_CHUNK_SIZE = 64 * 1024

# Retry status codes for which we will automatically retry the request
THROTTLE_RETRY_STATUS_CODES: set[int] = {429, 500, 502, 503, 504, 408}

# The attempts a ``raise_for_status=False`` client gets when it names no
# ``retry_max_attempts``. Three attempts wait 3-5 s in all (1 s, then 2 s, each
# plus up to 1 s of jitter): enough to ride out a short burst of 429s or a
# deploy's 502s, short enough that a block against a host that is down fails
# in seconds instead of holding its node, and the cap most callers that name
# one already chose. Each retry re-sends the request, POSTs included.
DEFAULT_RETRY_MAX_ATTEMPTS = 3

# Statuses meaning the request was not authenticated or not authorised. Both
# warrant asking the user to (re)connect, so they gate the setup-card paths.
AUTH_STATUS_CODES: set[int] = {401, 403}

# Only 401 unambiguously means "this credential was rejected". A bare 403 is
# routinely a scope decision or a WAF blocking our egress IP, both with a
# perfectly valid token — so anything destructive or blocking keys off this
# narrower set, never AUTH_STATUS_CODES.
CREDENTIAL_REJECTED_STATUS_CODES: set[int] = {401}

# List of IP networks to block
BLOCKED_IP_NETWORKS = [
    # --8<-- [start:BLOCKED_IP_NETWORKS]
    # IPv4 Ranges
    ipaddress.ip_network("0.0.0.0/8"),  # "This" Network
    ipaddress.ip_network("10.0.0.0/8"),  # Private-Use
    ipaddress.ip_network("100.64.0.0/10"),  # Shared Address Space (CGNAT, RFC 6598)
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("169.254.0.0/16"),  # Link Local
    ipaddress.ip_network("172.16.0.0/12"),  # Private-Use
    ipaddress.ip_network("192.0.0.0/24"),  # IETF Protocol Assignments
    ipaddress.ip_network("192.168.0.0/16"),  # Private-Use
    ipaddress.ip_network("198.18.0.0/15"),  # Benchmarking
    ipaddress.ip_network("224.0.0.0/4"),  # Multicast
    ipaddress.ip_network("240.0.0.0/4"),  # Reserved for Future Use
    # IPv6 Ranges
    ipaddress.ip_network("::/128"),  # Unspecified; Linux connects it to ::1
    ipaddress.ip_network("::1/128"),  # Loopback
    # Local-use NAT64 (RFC 8215): it translates to addresses inside some network,
    # and not always with the IPv4 address in the last 32 bits.
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("fc00::/7"),  # Unique local addresses (ULA)
    ipaddress.ip_network("fe80::/10"),  # Link-local
    ipaddress.ip_network("ff00::/8"),  # Multicast
    # --8<-- [end:BLOCKED_IP_NETWORKS]
]

ALLOWED_SCHEMES = ["http", "https"]
HOSTNAME_REGEX = re.compile(r"^[A-Za-z0-9.-]+$")  # Basic DNS-safe hostname pattern

# The longest name DNS allows, checked before IDNA encoding.
MAX_HOSTNAME_LENGTH = 253

# An allowlist of the text encodings a remote server may name for a response
# body we decode, by their canonical codec names (``codecs.lookup(...).name``),
# so any alias of a listed codec matches. Anything else is decoded as UTF-8.
ALLOWED_RESPONSE_CHARSETS = frozenset(
    {
        # Unicode
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "utf-32",
        "utf-32-le",
        "utf-32-be",
        "ascii",
        # ISO-8859, Windows, KOI8 and Mac single-byte families
        *(f"iso8859-{n}" for n in (*range(1, 12), *range(13, 17))),
        *(f"cp{n}" for n in range(1250, 1259)),
        "koi8-r",
        "koi8-u",
        "koi8-t",
        # DOS and other legacy single-byte codepages
        *(
            f"cp{n}"
            for n in (
                437,
                720,
                737,
                775,
                850,
                852,
                855,
                857,
                858,
                860,
                861,
                862,
                863,
                864,
                865,
                866,
                869,
                1125,
            )
        ),
        "kz1048",
        "ptcp154",
        "hp-roman8",
        "mac-arabic",
        "mac-croatian",
        "mac-cyrillic",
        "mac-farsi",
        "mac-greek",
        "mac-iceland",
        "mac-latin2",
        "mac-roman",
        "mac-romanian",
        "mac-turkish",
        # Thai
        "tis-620",
        "cp874",
        # Chinese
        "gb2312",
        "gbk",
        "gb18030",
        "big5",
        "big5hkscs",
        "cp950",
        # Japanese
        "shift_jis",
        "shift_jis_2004",
        "shift_jisx0213",
        "cp932",
        "euc_jp",
        "euc_jis_2004",
        "euc_jisx0213",
        "iso2022_jp",
        "iso2022_jp_1",
        "iso2022_jp_2",
        "iso2022_jp_2004",
        "iso2022_jp_3",
        "iso2022_jp_ext",
        # Korean
        "euc_kr",
        "cp949",
        "johab",
        "iso2022_kr",
    }
)

# IPv6 prefixes whose last 32 bits are an IPv4 address the network may route
# the connection to: the well-known NAT64 prefix (RFC 6052), SIIT (RFC 2765)
# and the deprecated IPv4-compatible form (RFC 4291).
_IPV4_IN_LOW_BITS_NETWORKS = [
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("::ffff:0:0:0/96"),
    ipaddress.ip_network("::/96"),
]


def _embedded_ipv4(ip_addr: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 address encodes, if it is a form that embeds one."""
    if ip_addr.ipv4_mapped:
        return ip_addr.ipv4_mapped
    if ip_addr.sixtofour:
        return ip_addr.sixtofour
    if ip_addr.teredo:
        # (server, client): the client is the host the packets are delivered to
        return ip_addr.teredo[1]
    if any(ip_addr in network for network in _IPV4_IN_LOW_BITS_NETWORKS):
        return ipaddress.IPv4Address(int(ip_addr) & 0xFFFFFFFF)
    return None


def _is_ip_blocked(ip: str) -> bool:
    """
    Checks if the IP address is in a blocked network.

    IPv6 addresses that embed an IPv4 address (IPv4-mapped ``::ffff:127.0.0.1``,
    6to4, Teredo, NAT64, SIIT and IPv4-compatible) are also checked as that IPv4
    address, so the IPv4 blocklist cannot be bypassed by encoding a private IPv4
    address as IPv6.
    """
    ip_addr = ipaddress.ip_address(ip)

    if any(ip_addr in network for network in BLOCKED_IP_NETWORKS):
        return True

    if isinstance(ip_addr, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(ip_addr)
        if embedded is not None:
            return any(embedded in network for network in BLOCKED_IP_NETWORKS)

    return False


# The headers a hop to another origin drops by name, lower-cased: HTTP header
# names are case-insensitive, so a caller passing ``{"authorization": ...}``
# must be stripped just like ``{"Authorization": ...}``. The hop also drops
# any header ``is_credential_header`` reads as a credential's; the names here
# are dropped whatever that comes to match.
SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        # API keys and tokens under their common names
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "x-auth-token",
        "x-access-token",
        "x-amz-security-token",
        # and under names some providers' APIs give them
        "key",
        "api_key",
        "profile-key",
        "x-access-key",
    }
)


def _drop_headers(headers: dict, names: Iterable[str]) -> None:
    """Remove every key in *names* from *headers*, matching case-insensitively.

    *names* is lower-cased here rather than by the caller: ``drop_headers`` is
    public on :meth:`Requests.request`, so a caller passing ``{"Authorization"}``
    would otherwise get a silent no-op and send the header anyway.
    """
    lowered = {n.lower() for n in names}
    for key in [k for k in headers if k.lower() in lowered]:
        headers.pop(key, None)


def _is_cross_origin(old_url: URL, new_url: URL) -> bool:
    """Whether a hop from *old_url* to *new_url* leaves the origin."""
    return (
        (old_url.scheme != new_url.scheme)
        or (old_url.hostname != new_url.hostname)
        or (old_url.port != new_url.port)
    )


def _remove_insecure_headers(
    headers: dict,
    cross_origin: bool,
    sensitive_headers: Iterable[str] = (),
) -> dict:
    """
    Removes, when *cross_origin*, every header that may carry a credential:
    those in ``SENSITIVE_HEADERS`` and *sensitive_headers*, and any whose name
    ``is_credential_header`` reads as a credential's, with ``token``,
    ``secret``, ``key``, ``auth`` or ``session`` (among others) anywhere in
    it. A hop that stays on the origin keeps every header.

    The rule also takes some names that carry no credential, such as
    ``Idempotency-Key``: on a hop to another origin, dropping one of those
    costs less than sending a credential the rule missed.

    The caller decides, so that one origin decision covers every carrier of a
    credential: these headers, and the ``auth=``/``cookies=`` kwargs aiohttp
    turns into headers of its own downstream of this call.
    """
    if cross_origin:
        _drop_headers(
            headers,
            [
                *SENSITIVE_HEADERS,
                *sensitive_headers,
                *(name for name in headers if is_credential_header(name)),
            ],
        )
    return headers


class HostResolver(abc.AbstractResolver):
    """
    A custom resolver that connects to specified IP addresses but still
    sets the TLS SNI to the original host name so the cert can match.
    """

    def __init__(self, ssl_hostname: str, ip_addresses: list[str]):
        self.ssl_hostname = ssl_hostname
        self.ip_addresses = ip_addresses
        self._default = aiohttp.ThreadedResolver()

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

    # AF_INET/AF_INET6 sockaddrs start with the address string; anything else is not
    # an IP we could validate or connect to.
    ip_list = [addr for info in infos if isinstance(addr := info[4][0], str)]
    ipv4 = [ip for ip in ip_list if ":" not in ip]
    ipv6 = [ip for ip in ip_list if ":" in ip]
    ip_addresses = ipv4 + ipv6

    if not ip_addresses:
        raise ValueError(f"No IP addresses found for {hostname}")
    return ip_addresses


async def validate_url_host(
    url: str, trusted_hostnames: Optional[list[str]] = None
) -> tuple[URL, bool, list[str]]:
    """
    Validates a (URL's) host string to prevent SSRF attacks by ensuring it does not
    point to a private, link-local, or otherwise blocked IP address — unless
    the hostname is explicitly trusted.

    Hosts in `trusted_hostnames` are permitted without checks.
    All other hosts are resolved and checked against `BLOCKED_IP_NETWORKS`.

    Params:
        url: A hostname, netloc, or URL to validate.
             If no scheme is included, `http://` is assumed.
        trusted_hostnames: A list of hostnames that don't require validation.

    Raises:
      ValueError:
        - if the URL has a disallowed URL scheme
        - if the URL/host string can't be parsed
        - if the hostname contains invalid or unsupported (non-ASCII) characters
        - if the host resolves to a blocked IP

    Returns:
        1. The validated, canonicalized, parsed host/URL,
           with hostname ASCII-safe encoding
        2. Whether the host is trusted (based on the passed `trusted_hostnames`).
        3. List of resolved IP addresses for the host; empty if the host is trusted.
    """
    parsed = parse_url(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed; allowed schemes: "
            f"{', '.join(ALLOWED_SCHEMES)}"
        )

    if not parsed.hostname:
        raise ValueError(f"Invalid host/URL; no host in parse result: {url}")

    if len(parsed.hostname.removesuffix(".")) > MAX_HOSTNAME_LENGTH:
        raise ValueError(f"Hostname is longer than {MAX_HOSTNAME_LENGTH} characters")

    # IDNA encode to prevent Unicode domain attacks
    try:
        ascii_hostname = idna.encode(parsed.hostname).decode("ascii")
    except idna.IDNAError:
        raise ValueError(f"Hostname '{parsed.hostname}' has unsupported characters")

    if not HOSTNAME_REGEX.match(ascii_hostname):
        raise ValueError(f"Hostname '{parsed.hostname}' has unsupported characters")

    # Re-create parsed URL object with IDNA-encoded hostname
    parsed = URL(
        parsed.scheme,
        (ascii_hostname if parsed.port is None else f"{ascii_hostname}:{parsed.port}"),
        quote(parsed.path, safe="/%:@"),
        parsed.params,
        parsed.query,
        parsed.fragment,
    )

    is_trusted = trusted_hostnames and any(
        matches_allowed_host(parsed, allowed)
        for allowed in (
            # Normalize + parse allowlist entries the same way for consistent matching
            parse_url(w)
            for w in trusted_hostnames
        )
    )

    if is_trusted:
        return parsed, True, []

    # If not allowlisted, go ahead with host resolution and IP target check
    return parsed, False, await resolve_and_check_blocked(ascii_hostname)


def matches_allowed_host(url: URL, allowed: URL) -> bool:
    if url.hostname != allowed.hostname:
        return False

    # Allow any port if not explicitly specified in the allowlist
    if allowed.port is None:
        return True

    return url.port == allowed.port


async def resolve_and_check_blocked(hostname: str) -> list[str]:
    """
    Resolves hostname to IPs and raises ValueError if any resolve to
    a blocked network. Returns the list of resolved IP addresses.
    """
    ip_addresses = await _resolve_host(hostname)
    for ip_str in ip_addresses:
        if _is_ip_blocked(ip_str):
            raise ValueError(
                f"Access to blocked or private IP address {ip_str} "
                f"for hostname {hostname} is not allowed."
            )
    return ip_addresses


def parse_url(url: str) -> URL:
    """Canonicalizes and parses a URL string."""
    url = url.lstrip("/ ").rstrip(" ").replace("\\", "/")

    # Ensure scheme is present for proper parsing.
    # Avoid regex to sidestep CodeQL py/polynomial-redos on user-controlled
    # input.  We only need to detect "scheme://"; urlparse() and the
    # ALLOWED_SCHEMES check downstream handle full validation.
    if "://" not in url:
        url = f"http://{url}"

    return urlparse(url)


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
            encoding = _response_charset(self.headers.get("content-type", ""))
        return self.content.decode(encoding or "utf-8", errors="replace")

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def raise_for_status(self) -> None:
        """Raise the same error ``Requests(raise_for_status=True)`` would have.

        For callers that need to inspect an error body before deciding what a
        status means, and only then want the standard exception.
        """
        if self.status >= 400:
            raise http_status_error(self.status, self.reason, self.content)


def _response_charset(content_type: str) -> str | None:
    """The allowed codec a ``Content-Type`` header names, or None."""
    match = re.search(r"charset=([^\s;]+)", content_type, flags=re.I)
    if not match:
        return None
    try:
        name = codecs.lookup(match.group(1).strip("\"'")).name
    except LookupError:
        return None
    return name if name in ALLOWED_RESPONSE_CHARSETS else None


def http_status_error(status: int, reason: str | None, body: bytes) -> Exception:
    """Build the exception ``Requests`` raises for an HTTP error status."""
    message = f"HTTP {status} Error: {reason}, Body: {body.decode(errors='replace')}"
    if 400 <= status <= 499:
        return HTTPClientError(message, status, body)
    if 500 <= status <= 599:
        return HTTPServerError(message, status, body)
    return Exception(message)


def _is_throttled(response: Response) -> bool:
    return response.status in THROTTLE_RETRY_STATUS_CODES


class _RetryBudget:
    """The throttled sends one request may make, across its redirect chain.

    A send that meets a throttle status counts against it. Until the count
    reaches ``attempts``, that send is made again after a jittered
    exponential backoff (1 s, then 2 s, 4 s and so on, each plus up to 1 s,
    and at most ``max_wait``); the last one is returned or raised. So a lone
    URL gets ``attempts`` sends, and a redirect chain makes at most that many
    throttled sends in all, however many hops it has.

    It waits with this module's own ``sleep``, which a test patches to note
    the waits instead of sleeping them.
    """

    def __init__(self, attempts: int, max_wait: float):
        self._attempts = attempts
        self._max_wait = max_wait
        self._spent = 0

    async def send_again(self) -> bool:
        """Count a throttled send; if another is allowed, wait out its backoff."""
        self._spent += 1
        if self._spent >= self._attempts:
            return False
        # The doubling stops at 2 ** 30 s, past any cap: from the 1,025th
        # retry on, 2 ** n is too large to add to a float.
        backoff = 2 ** min(self._spent - 1, 30) + random.uniform(0, 1)
        await sleep(max(0.0, min(backoff, self._max_wait)))
        return True


class _Redirect(BaseModel):
    """Where a hop's redirect leads, and what the next hop sends with it.

    Frozen, and built with ``model_construct`` from values this module made,
    so a hop neither validates nor copies it; ``_next_hop`` gives it copies of
    the headers and kwargs of its own."""

    model_config = ConfigDict(frozen=True)

    url: str
    method: str
    # False once the redirect has turned the request into a GET: the body is
    # then left behind for the rest of the chain.
    sends_body: bool
    headers: dict[str, Any]
    drop_headers: frozenset[str]
    kwargs: dict[str, Any]


# The headers that describe a request's body besides the ``Content-*`` ones
# (``Content-Type``, ``Content-Length``, ``Content-Encoding`` and the rest):
# they go with the body when a redirect leaves it behind.
_BODY_HEADERS = frozenset(
    {"transfer-encoding", "expect", "trailer", "digest", "repr-digest"}
)
# aiohttp options that frame a body, which a GET without one must not get:
# ``chunked`` would send a body-less GET a chunked body's terminator, which
# servers read as the start of the next request, and ``expect100`` asks for a
# ``100 Continue`` that RFC 9110 §10.1.1 forbids without content.
_BODY_KWARGS = ("chunked", "expect100")


def _next_hop(
    method: str,
    status: int,
    url: URL,
    location: str,
    *,
    headers: dict,
    drop_headers: frozenset[str],
    sensitive_headers: Iterable[str],
    kwargs: dict[str, Any],
    keep_post_on_same_origin: bool,
) -> _Redirect:
    """The hop a redirect from *url* to *location* leads to, and what it sends.

    A 303, and a 301 or 302 answering a POST, makes it a GET without the
    body, the headers that describe the body or the aiohttp options that
    frame it; with *keep_post_on_same_origin*, a 301 or 302 that stays on
    the origin keeps the POST. A hop to another origin also leaves the
    credentials and the ``Host`` behind.
    """
    # Joined onto the URL just requested, so a relative redirect resolves;
    # parsed with ``parse_url`` because the next hop parses it that way, and
    # only that folds a backslash in the authority into a slash, naming the
    # host that is connected to.
    redirect_url = parse_url(urljoin(url.geturl(), location))
    # Decided once, for every carrier of the credential below.
    cross_origin = _is_cross_origin(url, redirect_url)
    # The caller's opt-in: a 301 or 302 that stays on the origin keeps a POST.
    keeps_post = keep_post_on_same_origin and not cross_origin and status in (301, 302)
    drops_body = _drops_body(method, status) and not keeps_post
    if drops_body and method.upper() != "GET":
        # The origins only: a URL's path or query may hold a credential.
        logger.info(
            "A %s redirect sent a %s on as a GET without its body, from %s to %s",
            status,
            method.upper(),
            _origin(url),
            _origin(redirect_url),
        )
    new_headers = _next_hop_headers(
        headers, sensitive_headers, cross_origin=cross_origin, drops_body=drops_body
    )
    return _Redirect.model_construct(
        url=redirect_url.geturl(),
        method="GET" if drops_body else method,
        sends_body=not drops_body,
        headers=new_headers,
        # A dropped header stays dropped for the rest of the chain, though
        # ``extra_headers`` would put it back on every hop.
        drop_headers=drop_headers
        | {k.lower() for k in headers.keys() - new_headers.keys()},
        kwargs=_next_hop_kwargs(
            kwargs, cross_origin=cross_origin, drops_body=drops_body
        ),
    )


def _next_hop_headers(
    headers: dict,
    sensitive_headers: Iterable[str],
    *,
    cross_origin: bool,
    drops_body: bool,
) -> dict:
    """The headers of the hop a redirect leads to: without the credentials
    and the ``Host`` when it leaves the origin, and without the body's
    headers when it leaves the body behind."""
    new_headers = _remove_insecure_headers(
        dict(headers), cross_origin, sensitive_headers
    )
    if cross_origin:
        # The last hop's names another host. A pinned hop sets its own, and
        # aiohttp builds one from the URL for any other.
        _drop_headers(new_headers, ["host"])
    if drops_body:
        _drop_headers(new_headers, [k for k in new_headers if _describes_body(k)])
    return new_headers


def _next_hop_kwargs(
    kwargs: dict[str, Any], *, cross_origin: bool, drops_body: bool
) -> dict[str, Any]:
    """The aiohttp kwargs of the hop a redirect leads to.

    The hop goes to the URL its ``Location`` names, as other clients send it:
    aiohttp would add ``params=`` to that URL again, repeating a query the
    server kept and taking whatever it holds to another origin.

    ``auth=`` and ``cookies=`` are not headers at this point — aiohttp builds
    ``Authorization`` and ``Cookie`` from them per request, downstream of the
    header strip — so they go with the headers they stand in for on a hop to
    another origin. Same-origin hops keep them, as they keep those headers:
    an authenticated same-origin redirect is a normal flow.
    """
    next_kwargs = {name: value for name, value in kwargs.items() if name != "params"}
    if cross_origin:
        next_kwargs.pop("auth", None)
        next_kwargs.pop("cookies", None)
    if drops_body:
        for name in _BODY_KWARGS:
            next_kwargs.pop(name, None)
    return next_kwargs


def _drops_body(method: str, status: int) -> bool:
    """Whether a redirect with *status* makes the next hop a GET without the
    body, as RFC 9110 §15.4 has it: a 303 does for every method but HEAD, and
    a 301 or 302 does for a POST, as browsers and aiohttp's own redirects do.
    A 307 or 308, and a 301 or 302 answering any other method, sends the same
    method and body again."""
    if status == 303:
        return method.upper() != "HEAD"
    return status in (301, 302) and method.upper() == "POST"


def _origin(url: URL) -> str:
    """*url*'s scheme, host and port, without any user or password."""
    return f"{url.scheme}://{url.netloc.rpartition('@')[2]}"


def _describes_body(header: str) -> bool:
    name = header.lower()
    return name.startswith("content-") or name in _BODY_HEADERS


def _send_timeout(timeout: Any) -> float | None:
    """The seconds aiohttp gives one send in all, read from a request's
    ``timeout`` as aiohttp reads it: a ``ClientTimeout``'s total, or a number,
    where ``None`` and 0 mean no limit."""
    total = timeout.total if isinstance(timeout, aiohttp.ClientTimeout) else timeout
    return total if total is not None and total > 0 else None


class Requests:
    """
    A wrapper around an aiohttp ClientSession that validates URLs before
    making requests, preventing SSRF by blocking private networks and
    other disallowed address spaces.

    A response with a status in ``THROTTLE_RETRY_STATUS_CODES`` is retried,
    with jittered exponential backoff of at most ``retry_max_wait`` seconds
    between attempts. How many attempts a request gets, and what the caller
    gets when the last one still has a throttle status:

    ====================  ======================  ==============================
    ``raise_for_status``  ``retry_max_attempts``  Attempts, then
    ====================  ======================  ==============================
    ``False``             ``None``                ``DEFAULT_RETRY_MAX_ATTEMPTS``,
                                                  then the last response
    ``False``             ``n``                   ``n``, then the last response
    ``True`` (default)    ``None``                1, then ``HTTPClientError``
                                                  or ``HTTPServerError``
    ``True``              ``n``                   ``n``, then the last attempt's
                                                  ``HTTPClientError`` or
                                                  ``HTTPServerError``
    ====================  ======================  ==============================

    Any other status ends the request at once: it is raised with
    ``raise_for_status`` and returned without it. A connection error or a
    timeout is never retried. Only the hop that meets a throttle status is
    sent again, never the hops that redirected to it, and a redirect chain
    shares the attempts: it makes at most that many throttled sends in all.

    A redirect is followed as RFC 9110 §15.4 has it, and as browsers and
    aiohttp's own redirects do. After a 303, and after a 301 or 302 that
    answers a POST, the next hop is a GET without the body, the headers that
    describe it (``Content-Type``, ``Content-Length`` and the rest) or
    aiohttp's ``chunked`` and ``expect100``, and an ``info`` log names the
    two origins; a HEAD stays a HEAD. A 307 or 308, and a 301 or 302
    answering any other method, sends the same method and body again.
    ``keep_post_on_same_origin_redirect`` makes a 301 or 302 that stays on
    the origin keep a POST too, never a 303 or a hop to another origin: it is
    for a JSON-RPC endpoint, such as an MCP server's, that its server moves
    within its own origin, as nginx does to add a trailing slash.

    Every hop goes to the URL its ``Location`` names, resolved against the URL
    the last hop requested, its ``params=`` included, and is sent without
    ``params=`` again, through the same URL checks as the first. A hop to
    another origin goes without the ``auth=`` and ``cookies=`` it was given,
    its ``Host``, and any header that may carry a credential: those in
    ``SENSITIVE_HEADERS`` or ``sensitive_headers``, and any whose name
    ``is_credential_header`` reads as a credential's. A hop that stays on the
    origin keeps them all.

    A request that may send its body more than once, because it retries or
    follows redirects, sends all of it every time; ``prepare_body`` in
    ``backend.util.request_body`` sets out how. In short, a ``BytesIO`` or a
    ``StringIO``, as the body, an upload or a form's value, is rewound for
    each send, and any other file, which aiohttp closes once it has sent it,
    is read into memory once, as is an async iterable body. A body that has
    no way to be sent again is refused with a ``TypeError`` before the first
    send. A request that sends once, with one attempt and
    ``allow_redirects=False``, streams any body as it is.
    """

    def __init__(
        self,
        trusted_origins: list[str] | None = None,
        raise_for_status: bool = True,
        extra_url_validator: Callable[[URL], URL] | None = None,
        extra_headers: dict[str, str] | None = None,
        retry_max_wait: float = 300.0,
        retry_max_attempts: int | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        sensitive_headers: Iterable[str] = (),
        keep_post_on_same_origin_redirect: bool = False,
    ):
        self.max_response_bytes = max_response_bytes
        self.keep_post_on_same_origin_redirect = keep_post_on_same_origin_redirect
        # Headers bound to the request's origin under names neither
        # ``SENSITIVE_HEADERS`` nor ``is_credential_header`` knows: a
        # cross-origin redirect drops them too.
        self.sensitive_headers = frozenset(sensitive_headers)
        self.trusted_origins = []
        for url in trusted_origins or []:
            hostname = parse_url(url).netloc  # {host}[:{port}]
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

    def _max_attempts(self) -> int:
        """The attempts one request gets; the class docstring has the table."""
        if self.retry_max_attempts is not None:
            return self.retry_max_attempts
        return 1 if self.raise_for_status else DEFAULT_RETRY_MAX_ATTEMPTS

    def _retry_budget(self) -> _RetryBudget:
        """The throttled sends one request may make, and their waits."""
        return _RetryBudget(self._max_attempts(), self.retry_max_wait)

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict] = None,
        files: Uploads | None = None,
        data: Any | None = None,
        json: Any | None = None,
        allow_redirects: bool = True,
        max_redirects: int = 10,
        drop_headers: frozenset[str] = frozenset(),
        max_body_bytes: int | None = None,
        **kwargs,
    ) -> Response:
        """Send a request, following redirects through the same SSRF checks.

        ``max_body_bytes`` caps how much of a response body is read, counted
        after decompression; a longer body comes back cut to that many bytes
        plus one, so the caller can tell it was cut.

        A throttle status is retried, and a redirect followed, as the class
        docstring sets out.
        """
        if files is not None and json is not None:
            raise ValueError(
                "Cannot mix file uploads with JSON body; "
                "use 'data' for extra form fields instead."
            )
        request_body = await prepare_body(
            files,
            data,
            sends_again=self._max_attempts() > 1 or allow_redirects,
            check_url=lambda: self._checked_url(url),
            # The session's own timeout when the request names none.
            timeout=_send_timeout(kwargs.get("timeout", DEFAULT_TIMEOUT)),
        )
        return await self._hop(
            method,
            url,
            headers=headers,
            request_body=request_body,
            json=json,
            allow_redirects=allow_redirects,
            max_redirects=max_redirects,
            drop_headers=drop_headers,
            max_body_bytes=max_body_bytes,
            budget=self._retry_budget(),
            **kwargs,
        )

    async def _hop(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict],
        request_body: PreparedBody,
        json: Any | None,
        allow_redirects: bool,
        max_redirects: int,
        drop_headers: frozenset[str],
        max_body_bytes: int | None,
        budget: _RetryBudget,
        **kwargs,
    ) -> Response:
        """One hop: sent, and sent again on its own while it meets a throttle
        status and the budget lasts, then the hop its redirect names. A hop
        is never sent twice for a later hop's sake, and a hop its redirect
        turned into a GET sends no body, so nothing reads it again."""
        while True:
            try:
                hop = await self._request(
                    method,
                    url,
                    headers=headers,
                    request_body=request_body,
                    json=json,
                    allow_redirects=allow_redirects,
                    max_redirects=max_redirects,
                    drop_headers=drop_headers,
                    max_body_bytes=max_body_bytes,
                    **kwargs,
                )
            except (HTTPClientError, HTTPServerError) as error:
                if (
                    error.status_code in THROTTLE_RETRY_STATUS_CODES
                    and await budget.send_again()
                ):
                    continue
                raise
            if not isinstance(hop, _Redirect) and _is_throttled(hop):
                if await budget.send_again():
                    continue
            break
        if not isinstance(hop, _Redirect):
            return hop
        return await self._hop(
            hop.method,
            hop.url,
            headers=hop.headers,
            request_body=(
                request_body if hop.sends_body else PreparedBody.as_it_is(None, None)
            ),
            json=json if hop.sends_body else None,
            allow_redirects=allow_redirects,
            max_redirects=max_redirects - 1,
            drop_headers=hop.drop_headers,
            max_body_bytes=max_body_bytes,
            budget=budget,
            **hop.kwargs,
        )

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict],
        request_body: PreparedBody,
        json: Any | None,
        allow_redirects: bool,
        max_redirects: int,
        drop_headers: frozenset[str],
        max_body_bytes: int | None,
        **kwargs,
    ) -> Response | _Redirect:
        """One send of one hop: its response, or the redirect it names."""
        # Convert auth tuple to aiohttp.BasicAuth if necessary
        if "auth" in kwargs and isinstance(kwargs["auth"], tuple):
            kwargs["auth"] = aiohttp.BasicAuth(*kwargs["auth"])

        parsed_url, is_trusted, ip_addresses = await self._checked_url(url)

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
        session_kwargs: dict = {}
        if connector:
            session_kwargs["connector"] = connector

        # Merge any extra headers
        req_headers = dict(headers) if headers else {}
        if self.extra_headers is not None:
            req_headers.update(self.extra_headers)

        # A cross-origin redirect already stripped these; the merge above would
        # put them straight back, so a 302 to an attacker-controlled host would
        # replay the caller's bearer token. Drop them again, for every hop.
        _drop_headers(req_headers, drop_headers)

        # Set default User-Agent if not provided
        if "User-Agent" not in req_headers and "user-agent" not in req_headers:
            req_headers["User-Agent"] = DEFAULT_USER_AGENT

        # Override Host header if using IP connection
        if connector:
            req_headers["Host"] = hostname

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
                data=request_body.for_send(),
                json=json,
                **kwargs,
            ) as response:
                if self.raise_for_status:
                    try:
                        response.raise_for_status()
                    except ClientResponseError as e:
                        body = await _read_body(
                            response, self.max_response_bytes, max_body_bytes
                        )
                        raise http_status_error(
                            response.status, response.reason, body
                        ) from e

                # If allowed and a redirect is received, follow the redirect manually
                if allow_redirects and response.status in (301, 302, 303, 307, 308):
                    if max_redirects <= 0:
                        raise Exception("Too many redirects.")

                    location = response.headers.get("Location")
                    if not location:
                        return Response(
                            response=response,
                            url=original_url,
                            body=await _read_body(
                                response, self.max_response_bytes, max_body_bytes
                            ),
                        )

                    return _next_hop(
                        method,
                        response.status,
                        # Resolved against the URL as requested, with the
                        # query this send's ``params=`` added to it: a
                        # reference without a path, such as ``#result``,
                        # keeps that query.
                        parsed_url._replace(query=response.url.raw_query_string),
                        location,
                        headers=req_headers,
                        drop_headers=drop_headers,
                        sensitive_headers=self.sensitive_headers,
                        kwargs=kwargs,
                        keep_post_on_same_origin=self.keep_post_on_same_origin_redirect,
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
                    body=await _read_body(
                        response, self.max_response_bytes, max_body_bytes
                    ),
                )

    async def _checked_url(self, url: str) -> tuple[URL, bool, list[str]]:
        """*url* through the checks each send makes of it: the SSRF checks,
        which give its trust and its addresses, then the caller's own
        validator, which may refuse or rewrite it. A request whose body has
        to be read into memory makes them before reading it too, so a URL
        that either refuses reads nothing."""
        parsed_url, is_trusted, ip_addresses = await validate_url_host(
            url, self.trusted_origins
        )
        if self.extra_url_validator is not None:
            parsed_url = self.extra_url_validator(parsed_url)
        return parsed_url, is_trusted, ip_addresses

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


async def _read_body(
    response: ClientResponse, limit: int, cut_at: int | None = None
) -> bytes:
    """Buffer the body, refusing it as soon as it's known to be over ``limit``.

    With ``cut_at`` below the limit, the body is instead cut to that many
    bytes plus one, so the caller can tell it was cut, and never refused.
    Both read the decompressed stream in chunks and stop at their bound, so
    neither a long body nor a short compressed one that expands is held whole.
    """
    if cut_at is not None and cut_at < limit:
        return await _read_up_to(response, cut_at + 1)

    declared = response.content_length
    if declared is not None and declared > limit and _carries_a_body(response):
        response.close()
        raise ResponseTooLargeError(limit)
    body = await _read_up_to(response, limit + 1)
    if len(body) > limit:
        response.close()
        raise ResponseTooLargeError(limit)
    return body


def _carries_a_body(response: ClientResponse) -> bool:
    """A HEAD response, and a 1xx, 204, 205 or 304 one, declares the
    resource's Content-Length but transfers no body."""
    if response.method.upper() == "HEAD":
        return False
    return response.status >= 200 and response.status not in (204, 205, 304)


async def _read_up_to(response: ClientResponse, size: int) -> bytes:
    body = bytearray()
    async for chunk in response.content.iter_chunked(RESPONSE_READ_CHUNK_SIZE):
        body.extend(chunk[: size - len(body)])
        if len(body) >= size:
            break
    return bytes(body)
