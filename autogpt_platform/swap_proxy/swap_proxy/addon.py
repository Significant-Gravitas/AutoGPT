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
5. ``response``: known values are scrubbed back into placeholders.

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
from datetime import datetime, timezone
from typing import Optional

from mitmproxy import connection, http, tls
from mitmproxy.proxy import server_hooks
from mitmproxy.proxy.layers import modes

from swap_proxy.egress import EgressGuard
from swap_proxy.owners import Owner, OwnerDirectory
from swap_proxy.source import CredentialSource, SourceUnavailable
from swap_proxy.swap import (
    MAX_SCRUB_BYTES,
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

    async def request(self, flow: http.HTTPFlow) -> None:
        owner = self._owners.get(flow.client_conn)
        if owner is None:
            flow.kill()
            return
        request = flow.request
        names = placeholder_names(request)
        if not names:
            return
        host = request.pretty_host
        credentials = await self._credentials_for(flow, owner, host, names)
        swap = RequestSwap(credentials, host, request.method, request.path)
        swap.request(request)
        self._audit_events(owner, host, swap.events)

    async def websocket_message(self, flow: http.HTTPFlow) -> None:
        owner = self._owners.get(flow.client_conn)
        if owner is None or flow.websocket is None or not flow.websocket.messages:
            return
        message = flow.websocket.messages[-1]
        if message.is_text is False or not message.from_client:
            return
        try:
            text = message.content.decode("utf-8")
        except UnicodeDecodeError:
            return
        names = {m.group(1) for m in PLACEHOLDER_RE.finditer(text)}
        if not names:
            return
        host = flow.request.pretty_host
        credentials = await self._credentials_for(flow, owner, host, names)
        # No method and no path: a credential limited by either never swaps.
        swap = RequestSwap(credentials, host)
        new_text = swap.text(text)
        if new_text != text:
            message.content = new_text.encode("utf-8")
        self._audit_events(owner, host, swap.events)

    async def response(self, flow: http.HTTPFlow) -> None:
        owner = self._owners.get(flow.client_conn)
        response = flow.response
        user_id = owner.swap_user_id if owner else None
        if owner is None or user_id is None or response is None:
            return
        if response.stream or not self._provably_bound(flow):
            return
        if not is_scrubbable(response.headers.get("content-type", "")):
            return
        if len(response.raw_content or b"") > MAX_SCRUB_BYTES:
            return
        host = flow.request.pretty_host
        try:
            names = await self._source.bound_names(host)
            credentials = [
                c
                for name in names
                if (c := await self._source.resolve(user_id, name, host))
            ]
        except SourceUnavailable:
            return
        if not credentials:
            return
        try:
            text = response.get_text(strict=True)
        except ValueError:
            return
        if text is None:
            return
        scrubbed = scrub_text(text, credentials)
        if scrubbed != text:
            response.text = scrubbed

    # ------------------------------------------------------------ helpers

    def _provably_bound(self, flow: http.HTTPFlow) -> bool:
        """Is this request really going to the host it names?"""
        if self._allow_insecure_swap:
            return True
        if flow.request.scheme != "https":
            return False
        sni = flow.server_conn.sni
        return bool(sni) and sni.lower() == flow.request.pretty_host.lower()

    async def _credentials_for(
        self, flow: http.HTTPFlow, owner: Owner, host: str, names: set[str]
    ) -> dict[str, Credential]:
        """The owner's credentials among *names* that may go to *host*; every
        name left out is audited with the reason."""

        def refuse(reason: str, which: set[str]) -> dict[str, Credential]:
            self._audit_events(
                owner, host, [SwapEvent("refused", f"hsurr:{n}", reason) for n in which]
            )
            return {}

        if not self._provably_bound(flow):
            return refuse("unverified-destination", names)
        try:
            bound = await self._source.bound_names(host)
        except SourceUnavailable:
            return refuse("resolver-unavailable", names)
        # A placeholder on its way to a host it is not bound to: the one
        # signal that one went somewhere it should not.
        refuse("unbound-host", names - bound)
        wanted = names & bound
        user_id = owner.swap_user_id
        if user_id is None:
            return refuse("owner-does-not-swap", wanted)
        credentials: dict[str, Credential] = {}
        for name in sorted(wanted):
            try:
                credential = await self._source.resolve(user_id, name, host)
            except SourceUnavailable:
                refuse("resolver-unavailable", {name})
                continue
            if credential is None:
                refuse("not-connected", {name})
            else:
                credentials[name] = credential
        return credentials

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
