"""The swap itself: placeholders in a request become real values on the way out.

Ported from ``proxy/swap_addon.py`` in spark-vm, where it went through eight
review rounds; the rules below are that review's, kept as they were.  What
changed is only where credentials come from: there a secrets directory and a
registry file, here a set of ``Credential`` objects handed in per request, so
this module is pure and knows nothing of owners, Redis or the backend.

A placeholder is ``hsurr:<name>`` or ``hsurr:<name>:<entry>``; with no entry
it means ``access_token``.  It is replaced only when the credential is bound
to the request's host, and within that to its method and path if the
credential limits them.  A credential with no host binding never swaps.

Where it swaps.  By default in the ``Authorization`` header only, which is
where ``git`` (HTTP Basic, also for a token in the URL: git and curl turn URL
userinfo into that header, it never goes on the wire as part of the URL),
``gh`` (``token ...``) and ``curl -H`` / ``curl -u`` put a token.  A
placeholder anywhere else (another header, the path, the query, a body, a
websocket message) goes out literally and is audited as ``refused`` /
``outside-authorization``.  The reason is what a provider does with a request
body: a write-capable credential swapped into one can be stored there (a
gist, an issue, a blob) and read back later in any encoding the provider
offers (base64, hex, a git packfile), which no scrub of the response can
match.  A value in the ``Authorization`` header is used, not stored.

A credential that sets ``swap_anywhere`` (spark-vm's behaviour, and what the
rules below were written for) is swapped everywhere the swap looks, and how
the value is protected in each place:

- Headers.  ``Authorization: Basic`` is base64-decoded first (git and
  ``curl -u`` hide the placeholder inside it).  ``Referer`` and ``Origin`` are
  never touched: a swapped one would hand the value to the server's access log
  on every later request.  ``Cookie`` only for a credential that says so, and
  then without ``swap_anywhere`` too.
- Query string, per value, re-encoded.
- Path, per segment.  A segment whose decoded form did not change stays
  byte-identical, so existing escapes survive and a value cannot leave its
  segment.
- Body, by content type: JSON values are JSON-escaped, form fields are parsed
  and re-encoded (browser-encoded ``hsurr%3A`` included), other text is
  substituted plainly.  Binary bodies are left alone.

An entry named ``totp`` holds a base32 seed; the swap inserts the current
RFC 6238 code, never the seed.

Responses from a bound host are scrubbed: a known value in a text body is
replaced by its placeholder, so a page or API that echoes the key cannot hand
it back to the model through a text or screenshot read.  Values under eight
characters are never scrubbed (a one-character value would mangle the page);
TOTP codes only as whole tokens.  Images and binary bodies are a stated
residual, not a solved one.

Nothing here ever logs or returns a value: events carry names and reasons.

Not reachable in this deployment yet: ``swap_anywhere``, ``allowed_methods``,
``allowed_paths``, ``cookie``, ``no_scrub`` and TOTP entries.  The only
producer of a ``Credential`` (``source.py``) fills in a name, its values and
its hosts, so the swap outside ``Authorization``, the method and path limits,
the Cookie rule and the TOTP path keep their defaults.  They are ported and tested as they were and start to matter when
bindings become per-user (SECRT-2616, SECRT-2618).
"""

import base64
import binascii
import hashlib
import hmac
import json
import posixpath
import re
import struct
import time
import urllib.parse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

DEFAULT_ENTRY = "access_token"
TOTP_ENTRY = "totp"
PLACEHOLDER_RE = re.compile(r"hsurr:([A-Za-z0-9_-]+)(?::([A-Za-z0-9_-]+))?")
# As sent in application/x-www-form-urlencoded bodies (browsers encode ':').
ENCODED_PLACEHOLDER_RE = re.compile(
    r"hsurr%3A([A-Za-z0-9_-]+)(?:%3A([A-Za-z0-9_-]+))?", re.IGNORECASE
)
NEVER_SWAP_HEADERS = frozenset({"referer", "origin"})
MIN_SCRUB_LEN = 8
_SCRUBBABLE_TYPES = (
    "text/",
    "application/json",
    "application/javascript",
    "application/xml",
    "application/x-www-form-urlencoded",
)


@dataclass(frozen=True)
class Credential:
    """One credential of the connection's owner, as the swap needs it.

    *allowed_methods* and *allowed_paths* are static limits within the bound
    hosts: ``None`` means unrestricted, an empty tuple fails closed.
    *swap_anywhere* lets the value go outside the ``Authorization`` header
    (see the module docstring for why that is off); it arrives with the
    binding table (SECRT-2616), and nothing sets it yet.
    """

    name: str
    values: Mapping[str, str]
    allowed_hosts: tuple[str, ...]
    allowed_methods: Optional[tuple[str, ...]] = None
    allowed_paths: Optional[tuple[str, ...]] = None
    cookie: bool = False
    no_scrub: frozenset[str] = frozenset()
    swap_anywhere: bool = False

    def __repr__(self) -> str:  # a value must not reach a log through repr
        return f"Credential(name={self.name!r}, entries={sorted(self.values)})"


@dataclass(frozen=True)
class SwapEvent:
    """One audit fact: a placeholder that was swapped, or refused and why."""

    kind: Literal["swapped", "refused"]
    placeholder: str
    reason: str = ""


def host_in_list(host: Optional[str], entries: Iterable[str]) -> bool:
    """Exact names, or leading-dot entries for a domain's subdomains."""
    h = (host or "").lower().split(":")[0]
    for entry in entries:
        e = str(entry).lower()
        if h == e or (e.startswith(".") and h.endswith(e)):
            return True
    return False


def totp_code(seed: str, at: Optional[float] = None) -> str:
    """Current RFC 6238 code for a base32 seed: six digits, 30 s step, SHA-1."""
    s = seed.strip()
    key = base64.b32decode(s.upper() + "=" * (-len(s) % 8))
    counter = struct.pack(">Q", int(time.time() if at is None else at) // 30)
    digest = hmac.new(key, counter, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(code % 1_000_000).zfill(6)


def normalize_path(raw: str) -> str:
    """Percent-decode to a fixpoint, then resolve dot segments.

    ``/repos/../admin`` must not pass a ``/repos/`` prefix, ``%2e%2e`` must not
    smuggle dot segments past the check, and ``%252e%252e`` must not survive as
    ``%2e%2e`` for a server that decodes twice.
    """
    try:
        p = urllib.parse.urlsplit(raw).path
    except ValueError:
        p = raw or ""
    prev = None
    while p != prev:
        prev, p = p, urllib.parse.unquote(p)
    if not p.startswith("/"):
        p = "/" + p
    return posixpath.normpath(p) or "/"


def path_allowed(norm_path: str, prefixes: Iterable[str]) -> bool:
    """Segment-aligned: ``/repos/`` matches ``/repos/x``, not ``/repository``."""
    for prefix in prefixes:
        pp = normalize_path(str(prefix))
        if norm_path == pp or norm_path.startswith(pp.rstrip("/") + "/"):
            return True
    return False


def basic_decoded(value: str) -> Optional[str]:
    """``user:password`` out of an HTTP Basic header, or ``None`` if it is not one."""
    scheme, _, b64 = value.partition(" ")
    if scheme.lower() != "basic" or not b64.strip():
        return None
    try:
        return base64.b64decode(b64.strip(), validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None


def placeholder_names(
    request: Any, *, head: bool = True, body: bool = True
) -> set[str]:
    """Every credential name a request mentions, wherever a swap would look.

    Run before the swap so that only those credentials are fetched.  It looks
    inside Basic auth, the percent-encoded form and the path too: the same
    places the swap does, so nothing it would swap goes unfetched.  *head* is
    the request line and the headers, *body* the content.
    """
    chunks: list[str] = []
    if head:
        for key in request.headers.keys():
            for value in request.headers.get_all(key):
                chunks.append(value)
                if key.lower() == "authorization":
                    decoded = basic_decoded(value)
                    if decoded:
                        chunks.append(decoded)
        chunks.append(request.path)
        chunks.append(urllib.parse.unquote(request.path))
    if body and request.content:
        chunks.append(request.content.decode("utf-8", "ignore"))
    blob = "\n".join(chunks)
    names = {m.group(1) for m in PLACEHOLDER_RE.finditer(blob)}
    names |= {m.group(1) for m in ENCODED_PLACEHOLDER_RE.finditer(blob)}
    return names


@dataclass
class RequestSwap:
    """The swap for one request, against the owner's credentials for it."""

    credentials: Mapping[str, Credential]
    host: str
    method: Optional[str] = None
    path: Optional[str] = None
    events: list[SwapEvent] = field(default_factory=list)
    # ``(swapped, original)`` base64 of every HTTP Basic pair swapped: what a
    # server that echoes the header sends back, which no value matches.
    encoded: list[tuple[str, str]] = field(default_factory=list)

    def allows(self, credential: Credential) -> tuple[bool, str]:
        """Host binding first, then the credential's method and path limits."""
        if not host_in_list(self.host, credential.allowed_hosts):
            return False, "unbound-host"
        method = (self.method or "").upper()
        if credential.allowed_methods is not None:
            if method not in {m.upper() for m in credential.allowed_methods}:
                return False, "method-not-allowed"
        if credential.allowed_paths is not None:
            if self.path is None or method == "CONNECT":
                # Nothing to check the path against: inside a tunnel or a
                # websocket message a path-bound credential never swaps.
                return False, "path-not-verifiable"
            norm = normalize_path(self.path)
            if "%" in norm or ";" in norm or "\\" in norm:
                # After fixpoint decoding these are only smuggling shapes for
                # lenient servers: double-decode residue, path parameters,
                # backslash separators.
                return False, "path-not-allowed"
            if not path_allowed(norm, credential.allowed_paths):
                return False, "path-not-allowed"
        return True, ""

    def resolve(
        self, name: str, entry: str, *, outside_authorization: bool = True
    ) -> Optional[str]:
        """The value for a placeholder, or ``None`` to leave it as it is."""
        credential = self.credentials.get(name)
        if credential is None:
            return None
        ok, reason = self.allows(credential)
        if not ok:
            self._refuse(name, reason)
            return None
        if outside_authorization and not credential.swap_anywhere:
            self._refuse(name, "outside-authorization")
            return None
        if entry == TOTP_ENTRY:
            seed = credential.values.get(TOTP_ENTRY)
            if seed is None:
                return None
            try:
                return totp_code(seed)
            except (ValueError, binascii.Error):
                self._refuse(name, "bad-totp-seed")
                return None
        value = credential.values.get(entry)
        if value is None:
            # hsurr:github:8080 keeps its :8080; an unknown entry is not a
            # reason to swap the name alone.
            self._refuse(name, "unknown-entry")
        return value

    def _refuse(self, name: str, reason: str) -> None:
        event = SwapEvent("refused", f"hsurr:{name}", reason)
        if event not in self.events:
            self.events.append(event)

    def text(
        self,
        text: str,
        *,
        encode: Optional[Callable[[str], str]] = None,
        only: Optional[set[str]] = None,
        outside_authorization: bool = True,
    ) -> str:
        """Substitute placeholders; *encode* is applied to each value only,
        *only* limits which credentials may swap (the Cookie rule).  Outside
        the ``Authorization`` header (the default) only a credential with
        ``swap_anywhere`` swaps."""

        def repl(m: re.Match[str]) -> str:
            name, entry = m.group(1), m.group(2) or DEFAULT_ENTRY
            if only is not None and name not in only:
                if name in self.credentials:
                    self._refuse(name, "cookie-not-allowed")
                return m.group(0)
            value = self.resolve(
                name, entry, outside_authorization=outside_authorization
            )
            if value is None:
                return m.group(0)
            self.events.append(SwapEvent("swapped", m.group(0)))
            return encode(value) if encode else value

        return PLACEHOLDER_RE.sub(repl, text)

    def json_text(self, text: str) -> str:
        """Quotes and backslashes in a value cannot break the document."""
        return self.text(text, encode=lambda v: json.dumps(v)[1:-1])

    def form_body(self, text: str) -> str:
        """Parse the form, swap the values, re-encode: reserved characters in a
        value cannot corrupt the fields."""
        pairs = urllib.parse.parse_qsl(text, keep_blank_values=True)
        new_pairs = [(k, self.text(v)) for k, v in pairs]
        if new_pairs != pairs:
            return urllib.parse.urlencode(new_pairs)
        # Field names are never swapped, in either spelling.
        return text

    def basic_auth(self, value: str) -> str:
        decoded = basic_decoded(value)
        if decoded is None:
            return value
        new_decoded = self.text(decoded, outside_authorization=False)
        if new_decoded == decoded:
            return value
        swapped = base64.b64encode(new_decoded.encode()).decode("ascii")
        self.encoded.append((swapped, value.partition(" ")[2].strip()))
        return "Basic " + swapped

    def headers(self, request: Any) -> None:
        cookie_names = {n for n, c in self.credentials.items() if c.cookie}
        for key in list(request.headers.keys()):
            lowered = key.lower()
            if lowered in NEVER_SWAP_HEADERS:
                continue
            values = request.headers.get_all(key)
            if lowered == "authorization":
                new_values = [self.basic_auth(v) for v in values]
                # Not Basic (e.g. ``Bearer <placeholder>``): plain text.
                new_values = [
                    self.text(v, outside_authorization=False) if nv == v else nv
                    for v, nv in zip(values, new_values)
                ]
            elif lowered == "cookie":
                # ``cookie`` is an opt-in of its own.
                new_values = [
                    self.text(v, only=cookie_names, outside_authorization=False)
                    for v in values
                ]
            else:
                new_values = [self.text(v) for v in values]
            if new_values != values:
                request.headers.set_all(key, new_values)

    def request(self, request: Any) -> None:
        """Swap everywhere a request can carry a placeholder."""
        self.head(request)
        self.body(request)

    def head(self, request: Any) -> None:
        """Headers, query and path: everything that leaves before the body."""
        self.headers(request)
        query = list(request.query.items(multi=True))
        new_query = [(k, self.text(v)) for k, v in query]
        raw_path, mark, query_string = request.path.partition("?")
        raw_query = mark + query_string
        new_segments, path_changed = [], False
        for segment in raw_path.split("/"):
            decoded = urllib.parse.unquote(segment)
            swapped = self.text(decoded)
            if swapped != decoded:
                path_changed = True
                new_segments.append(urllib.parse.quote(swapped, safe=""))
            else:
                new_segments.append(segment)
        if path_changed or new_query != query:
            new_path = "/".join(new_segments) if path_changed else raw_path
            if new_query != query:
                new_path += "?" + urllib.parse.urlencode(new_query)
            else:
                # Untouched, so byte-identical: re-encoding it would change a
                # query that carries its own escaping, a signed URL's for one.
                new_path += raw_query
            request.path = new_path

    def body(self, request: Any) -> None:
        if not request.content:
            return
        try:
            body = request.content.decode("utf-8")
        except UnicodeDecodeError:
            return
        media = media_type(request.headers.get("content-type", ""))
        if media == "application/json" or media.endswith("+json"):
            new_body = self.json_text(body)
        elif media == "application/x-www-form-urlencoded":
            new_body = self.form_body(body)
        else:
            new_body = self.text(body)
        if new_body != body:
            request.content = new_body.encode("utf-8")


def media_type(content_type: Optional[str]) -> str:
    """``Application/JSON; charset=utf-8`` -> ``application/json``."""
    return (content_type or "").split(";")[0].strip().lower()


def is_scrubbable(content_type: Optional[str]) -> bool:
    c = media_type(content_type)
    if not c:
        return True  # unknown: try decoding, skip on failure
    return c.startswith(_SCRUBBABLE_TYPES) or c.endswith(("+json", "+xml"))


def scrub_replacements(
    credentials: Iterable[Credential],
) -> list[tuple[str, str, bool]]:
    """``(value, placeholder, whole_token)``, longest value first so that
    overlapping values replace correctly.

    A TOTP seed is never in the list; the current code is, because the model
    typed that placeholder into a page which may echo it back.  Codes match as
    whole tokens only: six digits collide with prices and ids.
    """
    triples: list[tuple[str, str, bool]] = []
    for credential in credentials:
        for entry, value in credential.values.items():
            suffix = "" if entry == DEFAULT_ENTRY else f":{entry}"
            placeholder = f"hsurr:{credential.name}{suffix}"
            if entry == TOTP_ENTRY:
                try:
                    triples.append((totp_code(value), placeholder, True))
                except (ValueError, binascii.Error):
                    pass
                continue
            if entry in credential.no_scrub or len(value or "") < MIN_SCRUB_LEN:
                continue
            triples.append((value, placeholder, False))
    triples.sort(key=lambda t: len(t[0]), reverse=True)
    return triples


def scrub_bytes(
    data: bytes,
    credentials: Iterable[Credential],
    encoded: Iterable[tuple[str, str]] = (),
) -> bytes:
    """``scrub_text`` on bytes, with every value as UTF-8 and as UTF-16 in
    either byte order, whatever charset the body declares.  For a body that
    does not decode as declared (one stray byte, an unknown charset name), and
    for one the box may read in another charset than the proxy did."""
    for swapped, original in encoded:
        data = data.replace(swapped.encode(), original.encode())
    for value, placeholder, whole_token in scrub_replacements(credentials):
        if whole_token:
            pattern = rb"(?<!\d)" + re.escape(value.encode()) + rb"(?!\d)"
            data = re.sub(pattern, placeholder.encode(), data)
            continue
        for codec in ("utf-8", "utf-16-le", "utf-16-be"):
            if (raw := value.encode(codec)) in data:
                data = data.replace(raw, placeholder.encode(codec))
    return data


def scrub_text(
    text: str,
    credentials: Iterable[Credential],
    encoded: Iterable[tuple[str, str]] = (),
) -> str:
    """Replace every known value in *text* with its placeholder, and every
    swapped Basic pair in *encoded* with the one the box sent."""
    for swapped, original in encoded:
        text = text.replace(swapped, original)
    for value, placeholder, whole_token in scrub_replacements(credentials):
        if whole_token:
            text = re.sub(r"(?<!\d)" + re.escape(value) + r"(?!\d)", placeholder, text)
        elif value in text:
            text = text.replace(value, placeholder)
    return text
