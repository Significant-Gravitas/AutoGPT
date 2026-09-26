"""Integration credential lookup with per-process TTL cache.

Provides token retrieval for connected integrations so that copilot tools
(e.g. bash_exec) can inject auth tokens into the execution environment without
hitting the database on every command.

Cache semantics (handled automatically by TTLCache):
- Token found → cached for _TOKEN_CACHE_TTL (60 s).  Avoids repeated DB hits
  for users who have credentials and are running many bash commands.
- No credentials found → cached for _NULL_CACHE_TTL (60 s).  Avoids a DB hit
  on every E2B command for users who haven't connected an account yet, while
  still picking up a newly-connected account within one minute.

Both caches are bounded to _CACHE_MAX_SIZE entries; cachetools evicts the
least-recently-used entry when the limit is reached.

Multi-worker note: the cached values are per-process, but invalidation is not.
The API server and the copilot executor each hold their own copy of these
caches, so a write in one cannot evict the other's; a subscription to the Redis
creds-changed bus does.  See ``_ensure_cache_invalidation_listener``.
"""

import asyncio
import logging
import threading
from collections.abc import Mapping
from typing import cast

from cachetools import TTLCache

from backend.copilot.providers import SUPPORTED_PROVIDERS
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.creds_events import listen_creds_changed
from backend.integrations.creds_manager import (
    IntegrationCredentialsManager,
    register_creds_changed_hook,
)
from backend.integrations.providers import ProviderName
from backend.util.retry import continuous_retry

logger = logging.getLogger(__name__)

# Derived from the single SUPPORTED_PROVIDERS registry for backward compat.
PROVIDER_ENV_VARS: dict[str, list[str]] = {
    slug: entry["env_vars"] for slug, entry in SUPPORTED_PROVIDERS.items()
}

# 60 s, not the original 300 s: the pub/sub invalidation below is best-effort
# (a Redis blip drops the message), so the TTL is the floor on how long a stale
# token can survive when it fails.  Five minutes was long enough for Otto
# to verify a re-authorization against the provider and report it as failed.
_TOKEN_CACHE_TTL = 60.0  # seconds — for found tokens
_NULL_CACHE_TTL = 60.0  # seconds — for "not connected" results
_CACHE_MAX_SIZE = 10_000


# Sentinel so ``pop`` keeps ``Cache.pop``'s "raise without a default" contract.
_MISSING = object()


class _LockedTTLCache(TTLCache):
    """TTLCache with a lock: the invalidation listener evicts entries from its
    own thread while copilot workers read them from theirs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with self._lock:
            super().__delitem__(key)

    def __contains__(self, key):
        with self._lock:
            return super().__contains__(key)

    def pop(self, key, default=_MISSING):
        with self._lock:
            if default is _MISSING:
                return super().pop(key)
            return super().pop(key, default)

    def popitem(self):
        with self._lock:
            return super().popitem()

    def pop_prefix(self, prefix: tuple) -> None:
        """Drop every entry whose key starts with *prefix*."""
        with self._lock:
            for key in [k for k in self.keys() if k[: len(prefix)] == prefix]:
                super().pop(key, None)


# (user_id, provider) → token string, or (user_id, provider, required_scopes)
# when the caller asked for specific scopes.  TTLCache handles expiry + eviction.
_CacheKey = tuple[str, str] | tuple[str, str, frozenset[str], str | None]
_token_cache: _LockedTTLCache = _LockedTTLCache(
    maxsize=_CACHE_MAX_SIZE, ttl=_TOKEN_CACHE_TTL
)
# Separate cache for "no credentials" results with a shorter TTL.
_null_cache: _LockedTTLCache = _LockedTTLCache(
    maxsize=_CACHE_MAX_SIZE, ttl=_NULL_CACHE_TTL
)

# GitHub user identity caches, keyed (user_id, credential_id): a user with two
# GitHub accounts has two identities, and a chat that picked one must not be
# served the other's. credential_id is None when no account was picked.
# Declared here so invalidate_user_provider_cache() can reference them.
_GH_IDENTITY_CACHE_TTL = 600.0  # 10 min — profile data rarely changes
_IdentityKey = tuple[str, str | None]
_gh_identity_cache: _LockedTTLCache = _LockedTTLCache(
    maxsize=_CACHE_MAX_SIZE, ttl=_GH_IDENTITY_CACHE_TTL
)
_gh_identity_null_cache: _LockedTTLCache = _LockedTTLCache(
    maxsize=_CACHE_MAX_SIZE, ttl=_NULL_CACHE_TTL
)


def _canonical_provider(provider: str) -> str:
    """``"ProviderName.GITHUB"`` -> ``"github"``.

    Credentials persisted under Python 3.13's ``str(StrEnum)`` carry the enum's
    repr as their provider, and change events pass it on as stored, while
    lookups (and so cache keys) use the canonical value.
    """
    if provider.startswith("ProviderName."):
        try:
            return ProviderName[provider.removeprefix("ProviderName.")].value
        except KeyError:
            pass
    return provider


def invalidate_user_provider_cache(user_id: str, provider: str) -> None:
    """Remove the cached entry for *user_id*/*provider* from both caches.

    Call this after storing new credentials so that the next
    ``get_provider_token()`` call performs a fresh DB lookup instead of
    serving a stale TTL-cached result.

    For GitHub specifically, also clears the git-identity caches so that
    ``get_github_user_git_identity()`` re-fetches the user's profile on
    the next call instead of serving stale identity data.
    """
    provider = _canonical_provider(provider)
    # Every scope-specific entry for this pair is stale too.
    _token_cache.pop_prefix((user_id, provider))
    _null_cache.pop_prefix((user_id, provider))

    if provider == "github":
        _gh_identity_cache.pop_prefix((user_id,))
        _gh_identity_null_cache.pop_prefix((user_id,))


# Same-process writes (a token refresh performed by this process) invalidate
# through the hook, without a Redis round trip.  Writes in other processes
# arrive over the bus instead.
try:
    register_creds_changed_hook(invalidate_user_provider_cache)
except RuntimeError:
    # Hook already registered (e.g. module re-import in tests).
    pass

# Module-level singleton to avoid re-instantiating IntegrationCredentialsManager
# on every cache-miss call to get_provider_token().
_manager = IntegrationCredentialsManager()


def _cache_key(
    user_id: str, provider: str, required: frozenset[str], credential_id: str | None
) -> _CacheKey:
    if required or credential_id:
        return (user_id, provider, required, credential_id)
    return (user_id, provider)


async def get_provider_token(
    user_id: str,
    provider: str,
    required_scopes: frozenset[str] = frozenset(),
    credential_id: str | None = None,
) -> str | None:
    """Return the user's access token for *provider*, or ``None`` if not connected.

    OAuth2 tokens are preferred (refreshed if needed); API keys are the fallback.
    Among several OAuth2 credentials, one granting every scope in
    *required_scopes* wins: that is the credential the connect card shows as
    connected, so injecting any other would send the model back to a card that
    already says "Connected".
    *credential_id* is the credential the user picked for this provider in the
    chat; when it is still stored, it is the only candidate.
    Both found tokens and "not connected" results are cached for 60 s, and a
    credential write in any process evicts the entry before that lapses.
    """
    _ensure_cache_invalidation_listener()
    cache_key = _cache_key(user_id, provider, required_scopes, credential_id)

    if cache_key in _null_cache:
        return None
    if cached := _token_cache.get(cache_key):
        return cached

    manager = _manager
    try:
        creds_list = await manager.store.get_creds_by_provider(user_id, provider)
    except Exception:
        logger.warning(
            "Failed to fetch %s credentials for user %s",
            provider,
            user_id,
            exc_info=True,
        )
        return None

    if credential_id is not None:
        # The user picked this one. If it is gone, that is "not connected",
        # never a reason to hand the sandbox another account's token.
        creds_list = [c for c in creds_list if c.id == credential_id]

    # Pass 1: prefer OAuth2 (carry scope info, refreshable via token endpoint).
    # Credentials covering the requested scopes come first, then ones with
    # "repo" (full git access, where a public-data-only token lacks push/pull).
    # The sort is stable, so ties keep their stored order, as the card does.
    # lock=False — background injection; not worth a distributed lock acquisition.
    def rank(creds: OAuth2Credentials) -> tuple[int, int]:
        granted = set(creds.scopes or [])
        return (
            0 if required_scopes <= granted else 1,
            0 if "repo" in granted else 1,
        )

    oauth2_creds = sorted(
        [cast(OAuth2Credentials, c) for c in creds_list if c.type == "oauth2"],
        key=rank,
    )
    refresh_failed = False
    for creds in oauth2_creds:
        if creds.type == "oauth2":
            try:
                fresh = await manager.refresh_if_needed(user_id, creds, lock=False)
                token = fresh.access_token.get_secret_value()
            except Exception:
                logger.warning(
                    "Failed to refresh %s OAuth token for user %s; "
                    "discarding stale token to force re-auth",
                    provider,
                    user_id,
                    exc_info=True,
                )
                # Do NOT fall back to the stale token — it is likely expired
                # or revoked.  Returning None forces the caller to re-auth,
                # preventing the LLM from receiving a non-functional token.
                refresh_failed = True
                continue
            # A refresh here publishes, and this process's own listener may evict
            # the entry just written; the cost is one extra lookup, not a leak.
            _token_cache[cache_key] = token
            return token

    # Pass 2: fall back to API key (no expiry, no refresh needed).
    for creds in creds_list:
        if creds.type == "api_key":
            token = cast(APIKeyCredentials, creds).api_key.get_secret_value()
            _token_cache[cache_key] = token
            return token

    # Only cache "not connected" when the user truly has no credentials for this
    # provider.  If we had OAuth credentials but refresh failed (e.g. transient
    # network error, event-loop mismatch), do NOT cache the negative result —
    # the next call should retry the refresh instead of being blocked for 60 s.
    if not refresh_failed:
        _null_cache[cache_key] = True
    return None


def _ensure_cache_invalidation_listener() -> None:
    """Subscribe this process to credential changes, once.

    Started from the cache read path so that every process holding a cache
    subscribes, and only those do — the in-process hook covers a write served
    by this process, and this covers the ones served elsewhere.
    """
    global _listener_thread
    with _listener_start_lock:
        if _listener_thread is not None:
            return
        _listener_thread = threading.Thread(
            target=lambda: asyncio.run(_consume_creds_changed_events()),
            name="creds-cache-invalidation",
            daemon=True,
        )
        _listener_thread.start()


@continuous_retry(retry_delay=5.0)
async def _consume_creds_changed_events() -> None:
    async for event in listen_creds_changed():
        invalidate_user_provider_cache(event.user_id, event.provider)
    raise RuntimeError("creds-changed subscription ended; resubscribing")


_listener_start_lock = threading.Lock()
_listener_thread: threading.Thread | None = None


async def get_integration_env_vars(
    user_id: str,
    required_scopes: Mapping[str, frozenset[str]] | None = None,
    selected: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return env vars for all providers the user has connected.

    Iterates :data:`PROVIDER_ENV_VARS`, fetches each token, and builds a flat
    ``{env_var: token}`` dict ready to pass to a subprocess or E2B sandbox.
    Only providers with a stored credential contribute entries.
    *required_scopes* maps a provider to the scopes its token should carry, and
    *selected* to the credential the user picked for it in this chat.
    """
    env: dict[str, str] = {}
    for provider, var_names in PROVIDER_ENV_VARS.items():
        if not var_names:
            continue
        scopes = (required_scopes or {}).get(provider, frozenset())
        token = await get_provider_token(
            user_id, provider, scopes, (selected or {}).get(provider)
        )
        if token:
            for var in var_names:
                env[var] = token
    return env


# ---------------------------------------------------------------------------
# GitHub user identity (for git committer env vars)
# ---------------------------------------------------------------------------


async def get_github_user_git_identity(
    user_id: str, credential_id: str | None = None
) -> dict[str, str] | None:
    """Fetch the GitHub user's name and email for git committer env vars.

    Uses the ``/user`` GitHub API endpoint with the user's stored token.
    Returns a dict with ``GIT_AUTHOR_NAME``, ``GIT_AUTHOR_EMAIL``,
    ``GIT_COMMITTER_NAME``, and ``GIT_COMMITTER_EMAIL`` if the user has a
    connected GitHub account.  Returns ``None`` otherwise.

    *credential_id* is the GitHub account the user picked in this chat, so the
    identity matches the token the sandbox was given for the same account.

    Results are cached for 10 minutes; "not connected" results are cached for
    60 s (same as null-token cache).
    """
    key: _IdentityKey = (user_id, credential_id)
    if key in _gh_identity_null_cache:
        return None
    if cached := _gh_identity_cache.get(key):
        return cached

    token = await get_provider_token(user_id, "github", frozenset(), credential_id)
    if not token:
        _gh_identity_null_cache[key] = True
        return None

    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "[git-identity] GitHub /user returned %s for user %s",
                        resp.status,
                        user_id,
                    )
                    return None
                data = await resp.json()
    except Exception as exc:
        logger.warning(
            "[git-identity] Failed to fetch GitHub profile for user %s: %s",
            user_id,
            exc,
        )
        return None

    name = data.get("name") or data.get("login") or "AutoGPT User"
    # GitHub may return email=null if the user has set their email to private.
    # Fall back to the noreply address GitHub generates for every account.
    email = data.get("email")
    if not email:
        gh_id = data.get("id", "")
        login = data.get("login", "user")
        email = f"{gh_id}+{login}@users.noreply.github.com"

    identity = {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
    }
    _gh_identity_cache[key] = identity
    return identity
