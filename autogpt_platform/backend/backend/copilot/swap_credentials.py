"""What the credential swap proxy asks the backend: bindings and values.

The proxy (``autogpt_platform/swap_proxy``) is a separate service with no
database access and no encryption key.  Every box egresses through it; when a
request carries a placeholder (``hsurr:github:<credential id>``) it asks here
for the value, naming the user the connection belongs to, the host the request
is going to and the box asking (its proxy credential).  The answer is only
what that box was granted, and only if the credential is bound to that host,
so the binding is enforced on this side of the wire as well as on the proxy's.

Lookup, OAuth refresh and cache invalidation are ``get_provider_token``'s, the
same path that fills ``GH_TOKEN`` today: there is no second store.

A credential's name is its provider slug, and its hosts come from
``SUPPORTED_PROVIDERS``: ``swap_hosts``, where the value may be sent, and
``content_hosts``, where what was stored with it is served back.  The proxy
opens both and scrubs responses from both; a credential it gets for either
names only the ``swap_hosts`` as the hosts it may be sent to, so the proxy's
own host check refuses to swap it into a request to a content host.
"""

from typing import Optional

from pydantic import BaseModel

from backend.copilot.integration_creds import get_provider_token, granted_to_box
from backend.copilot.providers import SUPPORTED_PROVIDERS, ProviderEntry
from backend.util.e2b_network import credential_record
from backend.util.exceptions import NotFoundError


class SwapCredential(BaseModel):
    """One credential as the proxy swaps it: entry name to value, and the
    only hosts it may be sent to."""

    name: str
    values: dict[str, str]
    allowed_hosts: list[str]


def _host_is_bound(host: str, allowed_hosts: list[str]) -> bool:
    """The same rule as ``host_in_list`` in ``swap_proxy/swap_proxy/swap.py``.

    Written twice because the two packages cannot import each other (their
    dependencies conflict), and the two must never disagree: that is what
    "enforced on both sides" rests on.  ``swap_credentials_test.py`` and the
    proxy's ``swap_test.py`` hold both to one table of hosts.
    """
    h = host.lower().split(":")[0]
    return any(
        h == entry or (entry.startswith(".") and h.endswith(entry))
        for entry in (e.lower() for e in allowed_hosts)
    )


def _bound_hosts(entry: ProviderEntry) -> list[str]:
    """Every host the proxy opens for a provider: where its value may go and
    where it may come back from."""
    if not entry["swap_hosts"]:
        return []
    return [*entry["swap_hosts"], *entry["content_hosts"]]


async def get_swap_bindings() -> dict[str, list[str]]:
    """Credential name to bound hosts, with no values: what lets the proxy
    leave every other host's traffic alone without asking."""
    return {
        slug: hosts
        for slug, entry in SUPPORTED_PROVIDERS.items()
        if (hosts := _bound_hosts(entry))
    }


class NoLiveBox(NotFoundError):
    """The box the proxy names is not a live box of the user's that gets swaps."""


async def _box_record(box: str, user_id: str) -> dict:
    """The egress record of the box asking, which must be a live box of
    *user_id*'s that gets swaps.  The proxy says which box a request comes
    from; this is the backend's own record of that box.

    Raises rather than answering ``None``: ``None`` reads as "not connected",
    and the proxy would pass that box's responses on unscrubbed.  An error is
    what makes it refuse them, which is right for a connection that outlived
    its box's credential (rotated at every reconnect), and for any mismatch
    between what the proxy and the backend know.
    """
    record = await credential_record(box)
    if (
        record is None
        or record.get("user_id") != user_id
        or record.get("swaps") is not True
    ):
        raise NoLiveBox("no live box of this user's that gets swaps")
    return record


async def resolve_swap_credential(
    user_id: str, name: str, host: str, box: str
) -> Optional[SwapCredential]:
    """*user_id*'s credential called *name*, if it may be sent to *host*,
    for the box whose proxy credential is *box*.

    The values are the credentials of *name* granted to that box
    (``integration_creds.grant_to_box``), each under its id, which is what
    ``hsurr:<name>:<credential id>`` names.  A chat's box is granted the
    credential the chat would have been handed as a token (its pick, else the
    best match for the requested scopes); another of the user's accounts, or
    an id typed by hand, resolves to nothing.

    ``None`` for an unknown name, an unbound host, or nothing granted that
    still yields a token (disconnected since): in each case the placeholder
    goes out as it is.  :class:`NoLiveBox` if *box* is not a live box of
    *user_id*'s that gets swaps (a revoked or rotated credential, a block's
    box).  A failure to read the grants, or to look a token up or refresh it,
    raises too (``ProviderTokenUnavailable`` for the latter) rather than
    answering ``None``: the proxy scrubs responses against this answer, so a
    failure has to reach it as an outage (it refuses what it cannot scrub),
    not as "nothing to scrub".
    """
    entry = SUPPORTED_PROVIDERS.get(name)
    if entry is None or not _host_is_bound(host, _bound_hosts(entry)):
        return None
    record = await _box_record(box, user_id)
    sandbox_id = record.get("sandbox_id")
    granted = await granted_to_box(sandbox_id, name) if sandbox_id else set()
    values: dict[str, str] = {}
    for credential_id in sorted(granted):
        # The token cache's path, keyed by the credential: a lookup a
        # command's placeholder already warmed, not one per stored account.
        # Locked: concurrent resolves for two hosts must not both spend a
        # single-use refresh token, and this service runs on one event loop.
        token = await get_provider_token(
            user_id, name, credential_id=credential_id, strict=True, lock=True
        )
        if token:
            values[credential_id] = token
    if not values:
        return None
    return SwapCredential(
        name=name, values=values, allowed_hosts=list(entry["swap_hosts"])
    )
