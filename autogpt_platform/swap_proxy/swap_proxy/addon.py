"""The mitmproxy addon: owners at the door, the guard on the way out, the swap
in between.

One connection's life:

1. ``socks5_auth``: E2B's host presents the box's proxy credential.  It
   resolves to an owner or the connection is refused (``owners.py``).
2. ``server_connect``: the destination is resolved, checked against private
   address space and pinned (``egress.py``).
3. ``tls_clienthello``: if no credential is bound to the host being dialled,
   the connection is passed through untouched: no interception, no decryption.
   If the backend has never answered, a swapping owner's connection is opened
   anyway, so that what cannot be scrubbed is refused rather than tunnelled.
   Only hosts that can receive a credential are opened.  spark-vm intercepts
   everything; serving many users, we decrypt only what we must.
4. ``request``: placeholders are swapped for the owner's values (``swap.py``),
   fetched from the backend for this user and this host (``source.py``).
5. ``response``: known values are scrubbed back into placeholders.  Websocket
   messages get the same two steps, one per direction.

Bodies and streaming.  mitmproxy streams a body larger than ``MAX_BODY_BYTES``
instead of holding it, and a streamed message's head is on the wire before
``request`` / ``response`` fire, so those two hooks alone would swap too late
and scrub nothing.  ``requestheaders`` and ``responseheaders`` therefore decide
first, for any body not known to fit:

- The head of a request (headers, path, query) is swapped in ``requestheaders``,
  before anything is sent.  A large ``git push`` authenticates this way while
  its pack streams through.
- A text body of unknown length (chunked, or HTTP/2 without a length) is held
  back by ``BufferedBody`` up to ``MAX_BODY_BYTES`` and swapped or scrubbed
  whole.
- A text *response* that is, or turns out to be, larger than that is refused:
  the flow is killed and the refusal audited.  It is never passed on
  unscrubbed.
- A text *request* body larger than that goes out as it is, its placeholders
  literal, and the audit says so.  That is the safe direction: the request
  fails at the provider and nothing leaks.
- Binary bodies stream untouched in both directions; they are neither swapped
  nor scrubbed at any size.
- A held body with a ``Content-Encoding`` is decoded within
  ``MAX_DECODED_BYTES`` (``decode.py``), never through mitmproxy's unbounded
  ``.content``.  Past that, or in an encoding that cannot be decoded within a
  bound, a response is refused like one too large to scrub, and a request
  body goes out as it is, audited as not swapped.

The audit records a swap only for bytes that have not left yet.

What is scrubbed is the owner's current values for the host plus those
swapped into the flow itself.  When the backend cannot say what the owner's
values are, a text response is refused and a server websocket message is
dropped, for an owner who gets swaps: an empty answer is not an empty set.

A value is swapped only into a request that is provably going to the bound
host: the scheme is https, mitmproxy has verified the upstream certificate
(its default, never turned off here), and the ``Host`` the request names is
the name that certificate was verified for.  The SOCKS5 destination alone
proves nothing, since the box chooses it.

Known limit: a connection stays authenticated for as long as it stays open,
also after its box's credential was rotated by a reconnect.  The credential it
presented was valid and the box is the same box; new connections need the new
one.
"""

import ipaddress
import json
import logging
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Union

from mitmproxy import connection, http, tls
from mitmproxy.net.http.http1.read import expected_http_body_size
from mitmproxy.proxy import server_hooks
from mitmproxy.proxy.layers import modes
from OpenSSL import SSL

from swap_proxy.decode import DecodedTooLarge, Undecodable, bounded_decode
from swap_proxy.egress import EgressGuard
from swap_proxy.owners import Owner, OwnerDirectory
from swap_proxy.source import CredentialSource, SourceUnavailable
from swap_proxy.swap import (
    PLACEHOLDER_RE,
    Credential,
    RequestSwap,
    SwapEvent,
    is_scrubbable,
    placeholder_names,
    scrub_text,
)

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("swap_proxy.audit")

# The one size in this service.  Up to here a body is held in memory, so it can
# be swapped or scrubbed; mitmproxy streams anything larger
# (``stream_large_bodies`` is set to this very number in ``__main__``).  It
# bounds what one connection can make the proxy hold, and the synchronous swap
# and scrub work one message can put on the shared event loop.
MAX_BODY_BYTES = 5 * 1024 * 1024
# What a held body may decode to.  ``MAX_BODY_BYTES`` counts bytes on the wire,
# and a compressed body stands for more: ordinary JSON or HTML shrinks to a
# fifth or less, a bomb to a thousandth.  Four times the wire limit lets most
# honest compressed bodies through and keeps one message's memory, and its
# decode-swap-scrub-encode time on the shared event loop, within a small
# multiple of what an unencoded body costs.  Enforced while decoding
# (``decode.py``), never by measuring a body that was already decoded.
MAX_DECODED_BYTES = 4 * MAX_BODY_BYTES

_HEAD_SWAPPED = "swap_proxy_head_swapped"
# The credentials whose values went into this flow, by name.
_SWAPPED = "swap_proxy_swapped"


class BufferedBody:
    """A ``stream`` callable that holds a body back and releases it transformed.

    mitmproxy calls it with every chunk and once more with ``b""`` at the end.
    Up to *limit* bytes are kept and nothing is forwarded; at the end
    *transform* gets the whole body and its result is sent.  Past the limit,
    or if data follows the end marker (an empty chunk mid-body would otherwise
    split a value across two transforms), *overflow* is told why, once.  What
    it returns decides the rest: ``True`` releases what was held and lets the
    remainder through untouched, ``False`` forwards nothing more.

    It returns lists: an empty ``bytes`` would be sent as a zero-length chunk,
    which ends a chunked body.
    """

    def __init__(
        self,
        limit: int,
        transform: Callable[[bytes], bytes],
        overflow: Callable[[str], bool],
    ):
        self._limit = limit
        self._transform = transform
        self._overflow = overflow
        self._held: list[bytes] = []
        self._size = 0
        self._ended = False
        self._release: Optional[bool] = None  # set once overflowed

    def __call__(self, chunk: bytes) -> list[bytes]:
        if self._release is not None:
            return [chunk] if self._release and chunk else []
        if chunk and self._ended:
            return self._give_up("data-after-end", chunk)
        if not chunk:
            self._ended = True
            body, self._held, self._size = b"".join(self._held), [], 0
            out = self._transform(body) if body else b""
            return [out] if out else []
        self._size += len(chunk)
        if self._size > self._limit:
            return self._give_up("body-too-large", chunk)
        self._held.append(chunk)
        return []

    def _give_up(self, reason: str, chunk: bytes) -> list[bytes]:
        held, self._held = self._held, []
        self._release = self._overflow(reason)
        return [*held, chunk] if self._release else []


_UNAVAILABLE = "resolver-unavailable"


@dataclass
class _Lookup:
    """The owner's credentials for a host, and why each other name has none."""

    credentials: dict[str, Credential] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)
    # A reason that holds for every name, whatever it is.
    blanket: str = ""

    @property
    def unavailable(self) -> bool:
        """The backend could not say, for some name or all of them: what the
        owner has for this host is unknown, which is not the same as nothing."""
        return self.blanket == _UNAVAILABLE or _UNAVAILABLE in self.refused.values()

    def refusals(self, names: set[str]) -> list[SwapEvent]:
        return [
            SwapEvent(
                "refused",
                f"hsurr:{name}",
                self.blanket or self.refused.get(name, "unbound-host"),
            )
            for name in sorted(names - self.credentials.keys())
        ]


_NOT_READABLE = {
    DecodedTooLarge: "decoded-too-large",
    Undecodable: "undecodable-encoding",
}


def plain_copy(message: http.Message) -> http.Message:
    """*message* with its content encoding undone, within ``MAX_DECODED_BYTES``:
    itself if it has none, else a copy.  mitmproxy's own ``.content`` and
    ``.text`` decode without a bound, so an encoded body is only ever read
    through this.  Raises ``DecodedTooLarge`` or ``Undecodable``."""
    encoding = message.headers.get("content-encoding", "")
    if encoding.strip().lower() in ("", "none", "identity"):
        return message
    decoded = bounded_decode(message.raw_content or b"", encoding, MAX_DECODED_BYTES)
    plain = message.copy()
    del plain.headers["content-encoding"]
    plain.raw_content = decoded
    return plain


def put_back(message: http.Message, plain: http.Message) -> None:
    """Re-encode what was changed in a ``plain_copy`` into the message."""
    if plain is not message:
        message.content = plain.raw_content


def is_address(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]").split("%")[0])
    except ValueError:
        return False
    return True


def known_size(
    request: http.Request, response: Optional[http.Response]
) -> Optional[int]:
    """The body's length if it is fixed before the body arrives, else ``None``:
    chunked, read-until-close, HTTP/2 without a length, or a bad header."""
    message = response or request
    try:
        size = expected_http_body_size(request, response)
    except ValueError:
        return None
    if size is None or size < 0:
        return None
    if "content-length" not in message.headers and not request.http_version.startswith(
        "HTTP/1"
    ):
        return None
    return size


class SwapProxyAddon:
    def __init__(
        self,
        owners: OwnerDirectory,
        source: CredentialSource,
        guard: EgressGuard,
        *,
        allow_insecure_swap: bool = False,
    ):
        self._directory = owners
        self._source = source
        self._guard = guard
        # Tests only: lets a swap happen over plain http to a local upstream.
        self._allow_insecure_swap = allow_insecure_swap
        self._owners: weakref.WeakKeyDictionary[connection.Client, Owner] = (
            weakref.WeakKeyDictionary()
        )
        # Connections whose TLS handshake is to be failed (``tls_start_client``).
        self._refused_tls: weakref.WeakSet[connection.Client] = weakref.WeakSet()

    # ------------------------------------------------------------ the door

    async def socks5_auth(self, data: modes.Socks5AuthData) -> None:
        owner = await self._directory.authenticate(data.username, data.password)
        if owner is None:
            self._audit(None, "-", "refused-connection", reason="unknown-credential")
            return
        data.valid = True
        self._owners[data.client_conn] = owner

    # ------------------------------------------------------------ the way out

    async def server_connect(self, data: server_hooks.ServerConnectionHookData) -> None:
        address = data.server.address
        if not address:
            return
        owner = self._owners.get(data.client)
        if owner is None:
            data.server.error = "swap-proxy: connection has no owner"
            return
        host = address[0]
        verdict = await self._guard.check(host)
        if verdict.ip is None:
            self._audit(
                owner,
                host,
                "refused-egress",
                reason=verdict.refused or "",
                ip=verdict.refused_ip or "-",
            )
            data.server.error = f"swap-proxy: egress refused ({verdict.refused})"
            return
        # The address that was checked is the address that is dialled.  SNI is
        # unaffected: it comes from the client's hello, not from this.
        data.server.address = (verdict.ip, address[1])

    async def tls_clienthello(self, data: tls.ClientHelloData) -> None:
        sni = data.client_hello.sni
        owner = self._owners.get(data.context.client)
        swaps = owner is not None and owner.swap_user_id is not None
        if not sni:
            # Bindings are names, so with none nothing can be swapped in, and
            # nothing that comes back can be judged: a box that gets swaps
            # could reach a bound provider by address and read, unscrubbed,
            # a value it stored there through an earlier swap.  Refused for
            # such a box; anyone else's is passed through.
            if swaps:
                self._audit(owner, "-", "refused-connection", reason="no-sni")
                self._refused_tls.add(data.context.client)
            else:
                data.ignore_connection = True
            return
        try:
            bound = bool(await self._source.bound_names(sni))
        except SourceUnavailable:
            # No bindings table at all yet (a cold start with the backend
            # down): whether this host is bound is unknown.  For an owner who
            # gets swaps it is opened, so that the response hooks refuse what
            # they cannot scrub; passed through, a value stored there earlier
            # would reach the box unread.  Anyone else's is left alone.
            bound = swaps
        if not bound:
            data.ignore_connection = True

    def tls_start_client(self, data: tls.TlsData) -> None:
        """Runs after mitmproxy's own ``tlsconfig`` has built the connection,
        and swaps it for one with no certificate: the handshake fails with an
        alert and mitmproxy closes the connection, so a refused box sees a
        clean TLS error.  (Leaving ``ssl_conn`` empty instead, which mitmproxy
        also treats as a failure, leaves the box waiting.)"""
        if data.context.client in self._refused_tls:
            refused = SSL.Connection(SSL.Context(SSL.TLS_SERVER_METHOD))
            refused.set_accept_state()
            data.ssl_conn = refused

    # ------------------------------------------------------------ the swap

    async def requestheaders(self, flow: http.HTTPFlow) -> None:
        """Before anything is sent.  A body known to fit is left to ``request``;
        for any other the head is swapped here, because it leaves first."""
        owner = self._owners.get(flow.client_conn)
        if owner is None:
            flow.kill()
            return
        request = flow.request
        if (
            owner.swap_user_id is not None
            and request.scheme == "http"
            and is_address(request.pretty_host)
        ):
            # Plain http names no host a binding could match either: the
            # same case as TLS without SNI, refused for the same box.
            self._audit(owner, request.pretty_host, "refused-request", reason="no-sni")
            flow.kill()
            return
        size = known_size(request, None)
        if size is not None and size <= MAX_BODY_BYTES:
            return
        host = request.pretty_host
        flow.metadata[_HEAD_SWAPPED] = True
        names = placeholder_names(request, body=False)
        if names:
            lookup = await self._lookup(flow, owner, host, names)
            swap = RequestSwap(lookup.credentials, host, request.method, request.path)
            swap.head(request)
            self._record_swap(flow, owner, host, lookup, names, swap.events)
        if not is_scrubbable(request.headers.get("content-type", "")):
            return  # binary: streams through as it is
        # Every credential the body could name: it is not here to be read yet.
        lookup = await self._lookup(flow, owner, host, None)
        if not lookup.credentials:
            return
        if size is not None:
            self._audit(owner, host, "body-not-swapped", reason="body-too-large")
            return
        method, path = request.method, request.path

        def swap_body(body: bytes) -> bytes:
            whole = request.copy()
            whole.raw_content = body
            try:
                plain = plain_copy(whole)
            except (DecodedTooLarge, Undecodable) as e:
                too_large(_NOT_READABLE[type(e)])
                return body
            before = plain.raw_content
            swap = RequestSwap(lookup.credentials, host, method, path)
            swap.body(plain)
            named = placeholder_names(plain, head=False)
            self._record_swap(flow, owner, host, lookup, named, swap.events)
            if plain.raw_content == before:
                return body
            put_back(whole, plain)
            return whole.raw_content or b""

        def too_large(reason: str) -> bool:
            self._audit(owner, host, "body-not-swapped", reason=reason)
            return True  # out as it is, placeholders literal

        request.stream = BufferedBody(MAX_BODY_BYTES, swap_body, too_large)

    async def request(self, flow: http.HTTPFlow) -> None:
        owner = self._owners.get(flow.client_conn)
        if owner is None:
            if flow.killable:
                flow.kill()
            return
        request = flow.request
        if request.stream:
            # Streamed: its bytes have left.  ``requestheaders`` swapped what
            # could be swapped; a swap now would reach no wire, only the audit.
            return
        head = not flow.metadata.get(_HEAD_SWAPPED)
        host = request.pretty_host
        names = placeholder_names(request, head=head, body=False)
        plain: Optional[http.Message] = None
        try:
            plain = plain_copy(request)
            names |= placeholder_names(plain, head=False)
        except (DecodedTooLarge, Undecodable) as e:
            # Without decoding it nobody can say whether it names a credential.
            # It goes out as it is, which is the safe direction, and the audit
            # says so whenever this owner has anything that could have gone in.
            if (await self._lookup(flow, owner, host, None)).credentials:
                self._audit(
                    owner, host, "body-not-swapped", reason=_NOT_READABLE[type(e)]
                )
        if not names:
            return
        lookup = await self._lookup(flow, owner, host, names)
        swap = RequestSwap(lookup.credentials, host, request.method, request.path)
        if head:
            swap.head(request)
        if plain is not None:
            before = plain.raw_content
            swap.body(plain)
            if plain.raw_content != before:
                put_back(request, plain)
        self._record_swap(flow, owner, host, lookup, names, swap.events)

    async def websocket_message(self, flow: http.HTTPFlow) -> None:
        owner = self._owners.get(flow.client_conn)
        if owner is None or flow.websocket is None or not flow.websocket.messages:
            return
        message = flow.websocket.messages[-1]
        if message.is_text is False:
            return
        try:
            text = message.content.decode("utf-8")
        except UnicodeDecodeError:
            return
        host = flow.request.pretty_host
        if not message.from_client:
            # What the server says back is scrubbed like a response body.
            credentials = await self._scrub_credentials(flow, owner)
            if credentials is None:
                message.drop()
                self._audit(owner, host, "refused-message", reason=_UNAVAILABLE)
                return
            scrubbed = scrub_text(text, credentials)
            if scrubbed != text:
                message.content = scrubbed.encode("utf-8")
                self._audit(owner, host, "scrubbed")
            return
        names = {m.group(1) for m in PLACEHOLDER_RE.finditer(text)}
        if not names:
            return
        lookup = await self._lookup(flow, owner, host, names)
        # No method and no path: a credential limited by either never swaps.
        swap = RequestSwap(lookup.credentials, host)
        new_text = swap.text(text)
        if new_text != text:
            message.content = new_text.encode("utf-8")
        self._record_swap(flow, owner, host, lookup, names, swap.events)

    async def responseheaders(self, flow: http.HTTPFlow) -> None:
        """Before the body arrives: a text response that may echo a value is
        never left for mitmproxy to stream past the scrub."""
        owner = self._owners.get(flow.client_conn)
        response = flow.response
        if owner is None or response is None or flow.error:
            return
        if not is_scrubbable(response.headers.get("content-type", "")):
            return
        size = known_size(flow.request, response)
        if size is not None and size <= MAX_BODY_BYTES:
            return  # held whole by mitmproxy; ``response`` scrubs it
        credentials = await self._scrub_credentials(flow, owner)

        def refuse(reason: str) -> bool:
            self._refuse_response(flow, owner, reason)
            return False

        if credentials is None:
            refuse(_UNAVAILABLE)
            return
        if not credentials:
            return
        host = flow.request.pretty_host
        if size is not None:
            refuse("too-large-to-scrub")
            return

        def scrub_body(body: bytes) -> bytes:
            whole = response.copy()
            whole.raw_content = body
            try:
                if self._scrub(whole, credentials):
                    self._audit(owner, host, "scrubbed")
            except (DecodedTooLarge, Undecodable) as e:
                refuse(_NOT_READABLE[type(e)])
                return b""
            return whole.raw_content or b""

        response.stream = BufferedBody(
            MAX_BODY_BYTES, scrub_body, lambda _: refuse("too-large-to-scrub")
        )

    async def response(self, flow: http.HTTPFlow) -> None:
        owner = self._owners.get(flow.client_conn)
        response = flow.response
        if owner is None or response is None:
            return
        if response.stream:
            return  # binary, or already handled by ``responseheaders``
        if not response.raw_content or not is_scrubbable(
            response.headers.get("content-type", "")
        ):
            return
        credentials = await self._scrub_credentials(flow, owner)
        if credentials is None:
            self._refuse_response(flow, owner, _UNAVAILABLE)
            return
        try:
            if self._scrub(response, credentials):
                self._audit(owner, flow.request.pretty_host, "scrubbed")
        except (DecodedTooLarge, Undecodable) as e:
            # A body that cannot be read cannot be vouched for: the box could
            # decode what the proxy would not.  Not passed on.
            self._refuse_response(flow, owner, _NOT_READABLE[type(e)])

    # ------------------------------------------------------------ helpers

    def _provably_bound(self, flow: http.HTTPFlow) -> bool:
        """Is this request really going to the host it names?"""
        if self._allow_insecure_swap:
            return True
        if flow.request.scheme != "https":
            return False
        sni = flow.server_conn.sni
        return bool(sni) and sni.lower() == flow.request.pretty_host.lower()

    async def _scrub_credentials(
        self, flow: http.HTTPFlow, owner: Owner
    ) -> Optional[list[Credential]]:
        """Every value of the owner's that this host may have been sent, and
        every value swapped into this very flow (a credential rotated since
        is still scrubbed).

        ``None`` when the backend cannot say, for an owner who gets swaps.  An
        empty answer would pass the body on as if it had been scrubbed, and a
        value can come back in a response that did not ask for it: one the
        box had swapped into something the provider stores (a gist, an issue,
        a file) comes back from a later, placeholder-free read.  So the
        caller refuses what it cannot scrub.
        """
        # The host the request names and the one the connection named, since
        # the two need not agree (and over plain http nothing proves either).
        hosts = {flow.request.pretty_host.lower()}
        if flow.server_conn.sni:
            hosts.add(flow.server_conn.sni.lower())
        credentials: list[Credential] = []
        for host in sorted(hosts):
            lookup = await self._lookup(flow, owner, host, None, proof=False)
            if owner.swap_user_id is not None and lookup.unavailable:
                return None
            credentials += lookup.credentials.values()
        swapped: dict[str, Credential] = flow.metadata.get(_SWAPPED, {})
        return [*credentials, *swapped.values()]

    def _refuse_response(self, flow: http.HTTPFlow, owner: Owner, reason: str) -> None:
        self._audit(owner, flow.request.pretty_host, "refused-response", reason=reason)
        if flow.killable:
            flow.kill()

    def _record_swap(
        self,
        flow: http.HTTPFlow,
        owner: Owner,
        host: str,
        lookup: _Lookup,
        names: set[str],
        events: list[SwapEvent],
    ) -> None:
        """Audit a swap and remember which credentials went into the flow."""
        self._audit_events(owner, host, lookup.refusals(names) + events)
        swapped = flow.metadata.setdefault(_SWAPPED, {})
        for event in events:
            name = event.placeholder.split(":")[1]
            if event.kind == "swapped" and name in lookup.credentials:
                swapped[name] = lookup.credentials[name]

    @staticmethod
    def _scrub(message: Union[http.Response, http.Request], credentials) -> bool:
        """Scrub a whole text body in place; ``True`` if a value was in it.
        Raises ``DecodedTooLarge`` or ``Undecodable`` for a body it cannot read."""
        if not credentials or not message.raw_content:
            return False
        plain = plain_copy(message)
        try:
            text = plain.get_text(strict=True)
        except ValueError:
            return False
        if text is None:
            return False
        scrubbed = scrub_text(text, credentials)
        if scrubbed == text:
            return False
        plain.text = scrubbed
        put_back(message, plain)
        return True

    async def _lookup(
        self,
        flow: http.HTTPFlow,
        owner: Owner,
        host: str,
        names: Optional[set[str]],
        *,
        proof: bool = True,
    ) -> _Lookup:
        """The owner's credentials among *names* that may go to *host*, and the
        reason for every name left out.  ``None`` asks for all bound to it.

        *proof* is for a swap: only where the connection proves the host.  A
        scrub asks without it, since removing a value never sends one."""
        if proof and not self._provably_bound(flow):
            return _Lookup(blanket="unverified-destination")
        try:
            bound = await self._source.bound_names(host)
        except SourceUnavailable:
            return _Lookup(blanket=_UNAVAILABLE)
        # A placeholder on its way to a host it is not bound to is the one
        # signal that one went somewhere it should not: ``unbound-host``.
        wanted = bound if names is None else names & bound
        user_id = owner.swap_user_id
        if user_id is None:
            return _Lookup(refused=dict.fromkeys(wanted, "owner-does-not-swap"))
        lookup = _Lookup()
        for name in sorted(wanted):
            try:
                credential = await self._source.resolve(user_id, name, host)
            except SourceUnavailable:
                lookup.refused[name] = _UNAVAILABLE
                continue
            if credential is None:
                lookup.refused[name] = "not-connected"
            else:
                lookup.credentials[name] = credential
        return lookup

    def _audit_events(self, owner: Owner, host: str, events: list[SwapEvent]) -> None:
        for event in events:
            self._audit(
                owner,
                host,
                event.kind,
                placeholder=event.placeholder,
                reason=event.reason,
            )

    def _audit(
        self, owner: Optional[Owner], host: str, event: str, **fields: str
    ) -> None:
        """One line per fact.  Names and reasons; never a value."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "owner": owner.label if owner else None,
            "user_id": owner.user_id if owner else None,
            "sandbox_id": owner.sandbox_id if owner else None,
            "host": host,
            **{k: v for k, v in fields.items() if v},
        }
        audit_logger.info(json.dumps(record, sort_keys=True))
