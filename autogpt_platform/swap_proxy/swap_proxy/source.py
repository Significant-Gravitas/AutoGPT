"""Where credentials come from: the backend, asked at swap time.

The proxy holds no database access and no encryption key.  It asks the
backend's internal service two things (``backend/copilot/swap_credentials.py``):
which hosts any credential is bound to, with no values, and one user's
credential for one host.  The backend does the lookup, the OAuth refresh and
the binding check; what arrives here is only what the request in hand needs.

Both answers are cached briefly.  The backend already invalidates its own
token cache across processes, so the few seconds here are a burst damper (one
``git push`` is many requests), not a second source of truth: a rotated or
revoked credential is gone from the proxy within ``_CREDENTIAL_TTL``.

Anything but a clean answer means no swap.  The placeholder goes out as it is
and the request fails at the provider: loudly, and with nothing leaked.
"""

import logging
import time
from typing import Optional, Protocol

import httpx

from swap_proxy.swap import Credential, host_in_list

logger = logging.getLogger(__name__)

_BINDINGS_TTL = 300.0
# After a failed refresh the previous table is kept this long before the
# backend is asked again: each attempt can take the whole timeout, and asking
# on every lookup would add it to every request for as long as the outage lasts.
_BINDINGS_RETRY = 30.0
_CREDENTIAL_TTL = 15.0
_CACHE_MAX = 10_000
_TIMEOUT = httpx.Timeout(10.0, connect=3.0)


class SourceUnavailable(Exception):
    """The backend could not be asked; the caller must not swap."""


class CredentialSource(Protocol):
    async def bound_names(self, host: str) -> set[str]: ...

    async def resolve(
        self, user_id: str, name: str, host: str, box: str
    ) -> Optional[Credential]: ...


class BackendCredentialSource:
    def __init__(self, base_url: str, client: Optional[httpx.AsyncClient] = None):
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=_TIMEOUT)
        self._bindings: Optional[tuple[float, dict[str, tuple[str, ...]]]] = None
        self._credentials: dict[
            tuple[str, str, str, str], tuple[float, Optional[Credential]]
        ] = {}

    async def bindings(self) -> dict[str, tuple[str, ...]]:
        now = time.monotonic()
        if self._bindings and self._bindings[0] > now:
            return self._bindings[1]
        try:
            data = await self._call("get_swap_bindings", {})
            fresh = {str(n): tuple(str(h) for h in hosts) for n, hosts in data.items()}
        except (SourceUnavailable, AttributeError, TypeError):
            if self._bindings:
                # Bindings are a static table; a stale copy beats treating
                # every host as unbound while the backend restarts.
                logger.warning("Bindings refresh failed; keeping the previous table")
                self._bindings = (now + _BINDINGS_RETRY, self._bindings[1])
                return self._bindings[1]
            raise SourceUnavailable("bindings") from None
        self._bindings = (now + _BINDINGS_TTL, fresh)
        return fresh

    async def bound_names(self, host: str) -> set[str]:
        """The credentials that may ever be sent to *host*."""
        table = await self.bindings()
        return {name for name, hosts in table.items() if host_in_list(host, hosts)}

    async def resolve(
        self, user_id: str, name: str, host: str, box: str
    ) -> Optional[Credential]:
        """*user_id*'s credential *name* for *host*, asked for the box whose
        proxy credential is *box*: the backend answers only for a live box of
        that user's, with what was granted to that box."""
        key = (user_id, name, host.lower(), box)
        now = time.monotonic()
        cached = self._credentials.get(key)
        if cached and cached[0] > now:
            return cached[1]
        data = await self._call(
            "resolve_swap_credential",
            {"user_id": user_id, "name": name, "host": host, "box": box},
        )
        credential = None
        if data is not None:
            try:
                credential = Credential(
                    name=str(data["name"]),
                    values={str(k): str(v) for k, v in data["values"].items()},
                    allowed_hosts=tuple(str(h) for h in data["allowed_hosts"]),
                )
            except (KeyError, AttributeError, TypeError):
                raise SourceUnavailable("malformed credential") from None
        if len(self._credentials) >= _CACHE_MAX:
            self._credentials.clear()
        self._credentials[key] = (now + _CREDENTIAL_TTL, credential)
        return credential

    async def _call(self, method: str, body: dict):
        try:
            response = await self._client.post(f"{self._base_url}/{method}", json=body)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as e:
            # The exception's text can carry the URL, never a value.
            logger.warning("Backend call %s failed: %s", method, type(e).__name__)
            raise SourceUnavailable(method) from None

    async def aclose(self) -> None:
        await self._client.aclose()
