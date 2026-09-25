"""Single source of truth for copilot-supported integration providers.

Both :mod:`~backend.copilot.integration_creds` (env-var injection) and
:mod:`~backend.copilot.tools.connect_integration` (UI setup card) import from
here, eliminating the risk of the two registries drifting out of sync.
"""

from typing import TypedDict


class ProviderEntry(TypedDict):
    """Metadata for a supported integration provider.

    Attributes:
        name: Human-readable display name (e.g. "GitHub").
        env_vars: Environment variable names injected when the provider is
            connected (e.g. ``["GH_TOKEN", "GITHUB_TOKEN"]``).
        default_scopes: Default OAuth scopes requested when the agent does not
            specify any.
    """

    name: str
    env_vars: list[str]
    default_scopes: list[str]


def _is_github_oauth_configured() -> bool:
    """Return True if GitHub OAuth env vars are set.

    Uses a lazy import to avoid triggering ``Secrets()`` during module import,
    which can fail in environments where secrets are not yet loaded (e.g. tests,
    CLI tooling).
    """
    from backend.blocks.github._auth import GITHUB_OAUTH_IS_CONFIGURED

    return GITHUB_OAUTH_IS_CONFIGURED


def _is_stripe_link_oauth_configured() -> bool:
    """Lazy for the same reason as :func:`_is_github_oauth_configured`."""
    from backend.integrations.oauth.stripe_link_hosted import (
        STRIPE_LINK_HOSTED_OAUTH_IS_CONFIGURED,
    )

    return STRIPE_LINK_HOSTED_OAUTH_IS_CONFIGURED


# -- Registry ----------------------------------------------------------------
# Add new providers here.  Both env-var injection and the setup-card tool read
# from this single registry.

SUPPORTED_PROVIDERS: dict[str, ProviderEntry] = {
    # No env vars: a Link wallet token never enters the sandbox. Listed so the
    # agent can ask for a Link connection before a checkout.
    "stripe_link": {
        "name": "Stripe Link",
        "env_vars": [],
        "default_scopes": ["payment_methods.agentic", "userinfo:read"],
    },
    "github": {
        "name": "GitHub",
        "env_vars": ["GH_TOKEN", "GITHUB_TOKEN"],
        "default_scopes": ["repo"],
    },
}


def get_provider_auth_types(provider: str) -> list[str]:
    """Return the supported credential types for *provider* at runtime.

    OAuth types are only offered when the corresponding OAuth client env vars
    are configured.
    """
    if provider == "stripe_link":
        # Same rule as the Link blocks' credential field: `device_code` is how
        # the public client connects, `oauth2` what its grant is stored as.
        if _is_stripe_link_oauth_configured():
            return ["oauth2"]
        return ["oauth2", "device_code"]
    if provider == "github":
        if _is_github_oauth_configured():
            return ["api_key", "oauth2"]
        return ["api_key"]
    # Default for unknown/future providers — API key only.
    return ["api_key"]
