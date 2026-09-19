"""The mitmproxy addon: owners at the door, the guard on the way out, the swap
in between.

One connection's life:

1. ``socks5_auth``: E2B's host presents the box's proxy credential.  It
   resolves to an owner or the connection is refused (``owners.py``).
2. ``server_connect``: the destination is resolved, checked against private
   address space and pinned (``egress.py``).
3. ``tls_clienthello``: if no credential is bound to the host being dialled,
   the connection is passed through untouched: no interception, no decryption.
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

The audit records a swap only for bytes that have not left yet.

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

_HEAD_SWAPPED = "swap_proxy_head_swapped"


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


@dataclass
class _Lookup:
    """The owner's credentials for a host, and why each other name has none."""

    credentials: dict[str, Credential] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)
    # A reason that holds for every name, whatever it is.
    blanket: str = ""

    def refusals(self, names: set[str]) -> list[SwapEvent]:
        return [
            SwapEvent(
                "refused",
                f"hsurr:{name}",
                self.blanket or self.refused.get(name, "unbound-host"),
            )
            for name in sorted(names - self.credentials.keys())
        ]


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
        try:
            bound = bool(sni) and bool(await self._source.bound_names(sni or ""))
        except SourceUnavailable:
            # Nothing can be swapped without the backend; do not open traffic
            # there is no reason to read.
            bound = False
        if not bound:
            data.ignore_connection = True

    # ------------------------------------------------------------ the swap

    async def requestheaders(self, flow: http.HTTPFlow) -> None:
        """Before anything is sent.  A body known to fit is left to ``request``;
        for any other the head is swapped here, because it leaves first."""
        owner = self._owners.get(flow.client_conn)
        if owner is None:
            flow.kill()
            return
        request = flow.request
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
            self._audit_events(owner, host, lookup.refusals(names) + swap.events)
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
            swap = RequestSwap(lookup.credentials, host, method, path)
            swap.body(whole)
            named = placeholder_names(whole, head=False)
            self._audit_events(owner, host, lookup.refusals(named) + swap.events)
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
        names = placeholder_names(request, head=head)
        if not names:
            return
        host = request.pretty_host
        lookup = await self._lookup(flow, owner, host, names)
        swap = RequestSwap(lookup.credentials, host, request.method, request.path)
        if head:
            swap.head(request)
        swap.body(request)
        self._audit_events(owner, host, lookup.refusals(names) + swap.events)

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
        self._audit_events(owner, host, lookup.refusals(names) + swap.events)

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
        if not credentials:
            return
        host = flow.request.pretty_host

        def refuse(reason: str) -> bool:
            self._audit(owner, host, "refused-response", reason=reason)
            if flow.killable:
                flow.kill()
            return False

        if size is not None:
            refuse("too-large-to-scrub")
            return

        def scrub_body(body: bytes) -> bytes:
            whole = response.copy()
            whole.raw_content = body
            if self._scrub(whole, credentials):
                self._audit(owner, host, "scrubbed")
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
        if not is_scrubbable(response.headers.get("content-type", "")):
            return
        credentials = await self._scrub_credentials(flow, owner)
        if self._scrub(response, credentials):
            self._audit(owner, flow.request.pretty_host, "scrubbed")

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
    ) -> list[Credential]:
        """Every value of the owner's that this host may have been sent."""
        lookup = await self._lookup(flow, owner, flow.request.pretty_host, None)
        return list(lookup.credentials.values())

    @staticmethod
    def _scrub(message: Union[http.Response, http.Request], credentials) -> bool:
        """Scrub a whole text body in place; ``True`` if a value was in it."""
        if not credentials or not message.raw_content:
            return False
        try:
            text = message.get_text(strict=True)
        except ValueError:
            return False
        if text is None:
            return False
        scrubbed = scrub_text(text, credentials)
        if scrubbed == text:
            return False
        message.text = scrubbed
        return True

    async def _lookup(
        self, flow: http.HTTPFlow, owner: Owner, host: str, names: Optional[set[str]]
    ) -> _Lookup:
        """The owner's credentials among *names* that may go to *host*, and the
        reason for every name left out.  ``None`` asks for all bound to it."""
        if not self._provably_bound(flow):
            return _Lookup(blanket="unverified-destination")
        try:
            bound = await self._source.bound_names(host)
        except SourceUnavailable:
            return _Lookup(blanket="resolver-unavailable")
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
                lookup.refused[name] = "resolver-unavailable"
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
