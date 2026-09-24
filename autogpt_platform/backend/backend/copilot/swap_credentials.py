"""What the credential swap proxy asks the backend: bindings and values.

The proxy (``autogpt_platform/swap_proxy``) is a separate service with no
database access and no encryption key.  Every box egresses through it; when a
request carries a placeholder (``hsurr:github``) it asks here for the value,
naming the user the connection belongs to and the host the request is going
to.  The answer is the credential only if it is bound to that host, so the
binding is enforced on this side of the wire as well as on the proxy's.

Lookup, OAuth refresh and cache invalidation are ``get_provider_token``'s, the
same path that fills ``GH_TOKEN`` today: there is no second store.

A credential's name is its provider slug, and its hosts come from
``SUPPORTED_PROVIDERS``: that table is the binding registry until per-user
bindings exist (SECRT-2616, SECRT-2618).
"""

from typing import Optional

from pydantic import BaseModel

from backend.copilot.integration_creds import (
    get_provider_token,
    get_provider_tokens_by_credential,
)
from backend.copilot.providers import SUPPORTED_PROVIDERS


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


async def get_swap_bindings() -> dict[str, list[str]]:
    """Credential name to bound hosts, with no values: what lets the proxy
    leave every other host's traffic alone without asking."""
    return {
        slug: list(entry["swap_hosts"])
        for slug, entry in SUPPORTED_PROVIDERS.items()
        if entry["swap_hosts"]
    }


async def resolve_swap_credential(
    user_id: str, name: str, host: str
) -> Optional[SwapCredential]:
    """*user_id*'s credential called *name*, if it may be sent to *host*.

    ``None`` for an unknown name, an unbound host, or a user who has not
    connected the provider: in each case the placeholder goes out as it is.
    A failure to look the token up or refresh it raises
    (``ProviderTokenUnavailable``) rather than answering ``None``: the proxy
    scrubs responses against this answer, so a failure has to reach it as an
    outage (it refuses what it cannot scrub), not as "nothing to scrub".

    The value ``hsurr:<name>`` stands for is the user's default credential
    (``access_token``, what ``get_provider_token`` picks with no arguments).
    Every stored credential is also an entry under its own id, which is what
    ``hsurr:<name>:<credential id>`` stands for: a box is handed the credential
    the chat picked (``get_integration_placeholder_env``), and a user with two
    accounts must get the one they chose.
    """
    entry = SUPPORTED_PROVIDERS.get(name)
    if entry is None or not _host_is_bound(host, entry["swap_hosts"]):
        return None
    values = await get_provider_tokens_by_credential(user_id, name)
    if token := await get_provider_token(user_id, name, strict=True):
        values["access_token"] = token
    if not values:
        return None
    return SwapCredential(
        name=name, values=values, allowed_hosts=list(entry["swap_hosts"])
    )
