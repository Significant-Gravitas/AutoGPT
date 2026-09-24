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
        swap_hosts: The only hosts the credential swap proxy may send this
            provider's token to (exact names, or ``.example.com`` for every
            subdomain).  Empty means the token is never swapped in.  List the
            hosts that *take* the token, not every host the provider serves
            from: a redirect target with a signed URL needs none.
        content_hosts: Hosts that serve back what was stored with the
            credential (raw files, attachments, release assets) but never take
            the token.  The proxy opens their traffic too, and a value in a
            text response from one is turned back into its placeholder, so a
            token a user once committed or pasted cannot be read back through
            them; nothing is ever swapped into a request to one.  Same syntax
            as *swap_hosts*, and no host may be in both.

    Both lists are the provider's whole footprint as far as the proxy is
    concerned: a host in neither is passed through unread.  They are declared
    here, beside the provider's variables and scopes, because a provider is
    only ever handed to a box through this table.
    """

    name: str
    env_vars: list[str]
    default_scopes: list[str]
    swap_hosts: list[str]
    content_hosts: list[str]


def _is_github_oauth_configured() -> bool:
    """Return True if GitHub OAuth env vars are set.

    Uses a lazy import to avoid triggering ``Secrets()`` during module import,
    which can fail in environments where secrets are not yet loaded (e.g. tests,
    CLI tooling).
    """
    from backend.blocks.github._auth import GITHUB_OAUTH_IS_CONFIGURED

    return GITHUB_OAUTH_IS_CONFIGURED


# -- Registry ----------------------------------------------------------------
# Add new providers here.  Both env-var injection and the setup-card tool read
# from this single registry.

SUPPORTED_PROVIDERS: dict[str, ProviderEntry] = {
    "github": {
        "name": "GitHub",
        "env_vars": ["GH_TOKEN", "GITHUB_TOKEN"],
        "default_scopes": ["repo"],
        # The API, git over HTTPS, release asset uploads, and the two
        # GitHub-owned content hosts that do take the token: a private repo's
        # raw file (``Authorization: token ...`` to raw.githubusercontent.com)
        # and git over HTTPS to a gist (Basic auth to gist.github.com).
        "swap_hosts": [
            "github.com",
            "api.github.com",
            "uploads.github.com",
            "raw.githubusercontent.com",
            "gist.github.com",
        ],
        # objects., gist., media. (LFS) and the other user-content domains:
        # served from signed URLs, they never need the token, so they are only
        # scrubbed.  codeload.github.com is left out: it serves only archives,
        # which are binary and never scrubbed, so opening its traffic would
        # cost TLS termination and protect nothing.
        "content_hosts": [".githubusercontent.com"],
    },
}


def get_provider_auth_types(provider: str) -> list[str]:
    """Return the supported credential types for *provider* at runtime.

    OAuth types are only offered when the corresponding OAuth client env vars
    are configured.
    """
    if provider == "github":
        if _is_github_oauth_configured():
            return ["api_key", "oauth2"]
        return ["api_key"]
    # Default for unknown/future providers — API key only.
    return ["api_key"]
