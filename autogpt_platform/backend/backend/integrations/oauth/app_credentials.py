"""The OAuth app (client id and secret) a provider signs in with here.

A provider can declare OAuth and still have no app configured in this
deployment, and then nobody can finish its sign-in. The router refuses to
start such a flow, and the "can we connect this?" check must not count it;
both read the answer from here so they cannot disagree.
"""

import os

from backend.integrations.oauth import CREDENTIALS_BY_PROVIDER, HANDLERS_BY_NAME
from backend.integrations.providers import ProviderName, provider_key
from backend.util.settings import Settings

settings = Settings()


def oauth_client_credentials(provider: "ProviderName | str") -> tuple[str, str] | None:
    """``(client_id, client_secret)`` for ``provider``'s OAuth app, or None
    when it has no OAuth handler or its app is not configured here."""
    key = provider_key(provider)
    if key not in HANDLERS_BY_NAME:
        return None
    configured = CREDENTIALS_BY_PROVIDER.get(key)
    if configured and not configured.use_secrets:
        # SDK provider: its builder named the environment variables.
        client_id = _env(configured.client_id_env_var)
        client_secret = _env(configured.client_secret_env_var)
    else:
        client_id = getattr(settings.secrets, f"{key}_client_id", None)
        client_secret = getattr(settings.secrets, f"{key}_client_secret", None)
    if not (client_id and client_secret):
        return None
    return client_id, client_secret


def _env(name: str | None) -> str | None:
    return os.getenv(name) if name else None
