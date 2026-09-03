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
Credential writes happen in the API process while this cache lives in the
copilot executor, so a subscription to the Redis creds-changed bus drops the
affected entry wherever it is held.  See ``_ensure_cache_invalidation_listener``.
"""

import asyncio
import logging
import threading
from typing import cast

from cachetools import TTLCache

from backend.copilot.providers import SUPPORTED_PROVIDERS
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.creds_events import listen_creds_changed
from backend.integrations.creds_manager import (
    IntegrationCredentialsManager,
    register_creds_changed_hook,
)
from backend.util.retry import continuous_retry

logger = logging.getLogger(__name__)

# Derived from the single SUPPORTED_PROVIDERS registry for backward compat.
PROVIDER_ENV_VARS: dict[str, list[str]] = {
    slug: entry["env_vars"] for slug, entry in SUPPORTED_PROVIDERS.items()
}

# 60 s, not the original 300 s: the pub/sub invalidation below is best-effort
# (a Redis blip drops the message), so the TTL is the floor on how long a stale
# token can survive when it fails.  Five minutes was long enough for AutoPilot
# to verify a re-authorization against the provider and report it as failed.
_TOKEN_CACHE_TTL = 60.0  # seconds — for found tokens
_NULL_CACHE_TTL = 60.0  # seconds — for "not connected" results
_CACHE_MAX_SIZE = 10_000


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

    def pop(self, key, default=None):
        with self._lock:
            return super().pop(key, default)


# (user_id, provider) → token string.  TTLCache handles expiry + eviction.
_token_cache: TTLCache[tuple[str, str], str] = _LockedTTLCache(
    maxsize=_CACHE_MAX_SIZE, ttl=_TOKEN_CACHE_TTL
)
# Separate cache for "no credentials" results with a shorter TTL.
_null_cache: TTLCache[tuple[str, str], bool] = _LockedTTLCache(
    maxsize=_CACHE_MAX_SIZE, ttl=_NULL_CACHE_TTL
)

# GitHub user identity caches (keyed by user_id only, not provider tuple).
# Declared here so invalidate_user_provider_cache() can reference them.
_GH_IDENTITY_CACHE_TTL = 600.0  # 10 min — profile data rarely changes
_gh_identity_cache: TTLCache[str, dict[str, str]] = _LockedTTLCache(
    maxsize=_CACHE_MAX_SIZE, ttl=_GH_IDENTITY_CACHE_TTL
)
_gh_identity_null_cache: TTLCache[str, bool] = _LockedTTLCache(
    maxsize=_CACHE_MAX_SIZE, ttl=_NULL_CACHE_TTL
)


def invalidate_user_provider_cache(user_id: str, provider: str) -> None:
    """Remove the cached entry for *user_id*/*provider* from both caches.

    Call this after storing new credentials so that the next
    ``get_provider_token()`` call performs a fresh DB lookup instead of
    serving a stale TTL-cached result.

    For GitHub specifically, also clears the git-identity caches so that
    ``get_github_user_git_identity()`` re-fetches the user's profile on
    the next call instead of serving stale identity data.
    """
    key = (user_id, provider)
    _token_cache.pop(key, None)
    _null_cache.pop(key, None)

    if provider == "github":
        _gh_identity_cache.pop(user_id, None)
        _gh_identity_null_cache.pop(user_id, None)


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


async def get_provider_token(user_id: str, provider: str) -> str | None:
    """Return the user's access token for *provider*, or ``None`` if not connected.

    OAuth2 tokens are preferred (refreshed if needed); API keys are the fallback.
    Both found tokens and "not connected" results are cached for 60 s, and a
    credential write in any process evicts the entry before that lapses.
    """
    _ensure_cache_invalidation_listener()
    cache_key = (user_id, provider)

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

    # Pass 1: prefer OAuth2 (carry scope info, refreshable via token endpoint).
    # Sort so broader-scoped tokens come first: a token with "repo" scope covers
    # full git access, while a public-data-only token lacks push/pull permission.
    # lock=False — background injection; not worth a distributed lock acquisition.
    oauth2_creds = sorted(
        [c for c in creds_list if c.type == "oauth2"],
        key=lambda c: 0 if "repo" in (cast(OAuth2Credentials, c).scopes or []) else 1,
    )
    refresh_failed = False
    for creds in oauth2_creds:
        if creds.type == "oauth2":
            try:
                fresh = await manager.refresh_if_needed(
                    user_id, cast(OAuth2Credentials, creds), lock=False
                )
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
    """Start this process's subscription to credential changes, once.

    Started from the cache read path rather than from a service's startup: a
    subscription wired up somewhere else is the defect this fixes — the
    in-process hook was registered by a module the writing process never
    imported, so it silently invalidated nothing.  Anything that reads the
    cache now also subscribes to it.
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


async def get_integration_env_vars(user_id: str) -> dict[str, str]:
    """Return env vars for all providers the user has connected.

    Iterates :data:`PROVIDER_ENV_VARS`, fetches each token, and builds a flat
    ``{env_var: token}`` dict ready to pass to a subprocess or E2B sandbox.
    Only providers with a stored credential contribute entries.
    """
    env: dict[str, str] = {}
    for provider, var_names in PROVIDER_ENV_VARS.items():
        token = await get_provider_token(user_id, provider)
        if token:
            for var in var_names:
                env[var] = token
    return env


# ---------------------------------------------------------------------------
# GitHub user identity (for git committer env vars)
# ---------------------------------------------------------------------------


async def get_github_user_git_identity(user_id: str) -> dict[str, str] | None:
    """Fetch the GitHub user's name and email for git committer env vars.

    Uses the ``/user`` GitHub API endpoint with the user's stored token.
    Returns a dict with ``GIT_AUTHOR_NAME``, ``GIT_AUTHOR_EMAIL``,
    ``GIT_COMMITTER_NAME``, and ``GIT_COMMITTER_EMAIL`` if the user has a
    connected GitHub account.  Returns ``None`` otherwise.

    Results are cached for 10 minutes; "not connected" results are cached for
    60 s (same as null-token cache).
    """
    if user_id in _gh_identity_null_cache:
        return None
    if cached := _gh_identity_cache.get(user_id):
        return cached

    token = await get_provider_token(user_id, "github")
    if not token:
        _gh_identity_null_cache[user_id] = True
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
    _gh_identity_cache[user_id] = identity
    return identity
