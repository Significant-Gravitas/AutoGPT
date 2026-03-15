"""Integration credential lookup with per-process TTL cache.

Provides token retrieval for connected integrations so that copilot tools
(e.g. bash_exec) can inject auth tokens into the execution environment without
hitting the database on every command.

Cache semantics:
- Token found → cached for _TOKEN_CACHE_TTL (5 min).  Avoids repeated DB hits
  for users who have credentials and are running many bash commands.
- No credentials found → cached for _NULL_CACHE_TTL (60 s).  Avoids a DB hit
  on every E2B command for users who haven't connected an account yet, while
  still picking up a newly-connected account within one minute.
"""

import logging
import time
from typing import cast

from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.creds_manager import IntegrationCredentialsManager

logger = logging.getLogger(__name__)

# Maps provider slug → env var names to inject when the provider is connected.
# Add new providers here when adding integration support.
# NOTE: keep in sync with connect_integration._PROVIDER_INFO — both registries
# must be updated when adding a new provider.
PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "github": ["GH_TOKEN", "GITHUB_TOKEN"],
}

_TOKEN_CACHE_TTL = 300.0  # seconds — for found tokens
_NULL_CACHE_TTL = 60.0  # seconds — for "not connected" results
_SENTINEL = ""  # cached value meaning "no credentials found"
# (user_id, provider) → (token_or_sentinel, monotonic expiry)
_token_cache: dict[tuple[str, str], tuple[str, float]] = {}


async def get_provider_token(user_id: str, provider: str) -> str | None:
    """Return the user's access token for *provider*, or ``None`` if not connected.

    OAuth2 tokens are preferred (refreshed if needed); API keys are the fallback.
    Found tokens are cached for _TOKEN_CACHE_TTL (5 min).  "Not connected" results
    are cached for _NULL_CACHE_TTL (60 s) to avoid a DB hit on every bash_exec
    command for users who haven't connected yet, while still picking up a
    newly-connected account within one minute.
    """
    now = time.monotonic()
    cache_key = (user_id, provider)
    cached = _token_cache.get(cache_key)
    if cached is not None:
        token, expires_at = cached
        if now < expires_at:
            return token or None  # _SENTINEL ("") → None
        del _token_cache[cache_key]

    manager = IntegrationCredentialsManager()
    try:
        creds_list = await manager.store.get_creds_by_provider(user_id, provider)
    except Exception:
        logger.debug("Failed to fetch %s credentials for user %s", provider, user_id)
        return None

    # Pass 1: prefer OAuth2 (carry scope info, refreshable via token endpoint).
    # lock=False — background injection; not worth a distributed lock acquisition.
    for creds in creds_list:
        if creds.type == "oauth2":
            try:
                fresh = await manager.refresh_if_needed(
                    user_id, cast(OAuth2Credentials, creds), lock=False
                )
                token = fresh.access_token.get_secret_value()
            except Exception:
                logger.debug(
                    "Failed to refresh %s OAuth token for user %s", provider, user_id
                )
                token = cast(OAuth2Credentials, creds).access_token.get_secret_value()
            _token_cache[cache_key] = (token, now + _TOKEN_CACHE_TTL)
            return token

    # Pass 2: fall back to API key (no expiry, no refresh needed).
    for creds in creds_list:
        if creds.type == "api_key":
            token = cast(APIKeyCredentials, creds).api_key.get_secret_value()
            _token_cache[cache_key] = (token, now + _TOKEN_CACHE_TTL)
            return token

    # No credentials found — cache the sentinel to avoid repeated DB hits.
    _token_cache[cache_key] = (_SENTINEL, now + _NULL_CACHE_TTL)
    return None


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
