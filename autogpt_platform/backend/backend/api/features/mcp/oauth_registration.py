from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, model_validator

from backend.blocks.mcp.oauth import MCPTokenEndpointAuthMethod
from backend.util.settings import Secrets


class PreregisteredApp(BaseModel):
    # The ``Secrets`` fields holding the app's client id and secret.
    client_id_field: str
    client_secret_field: str
    # Hosts the discovered authorize and token endpoints must be on. The secret
    # is posted to the token endpoint, which comes from the server's metadata.
    endpoint_hosts: frozenset[str]


# MCP servers that refuse Dynamic Client Registration and only accept an OAuth
# app registered with them in advance, keyed by server host.
PREREGISTERED_APPS: dict[str, PreregisteredApp] = {
    "mcp.slack.com": PreregisteredApp(
        client_id_field="slack_mcp_client_id",
        client_secret_field="slack_mcp_client_secret",
        endpoint_hosts=frozenset({"slack.com"}),
    ),
}


class MCPClientRegistration(BaseModel):
    client_id: str = Field(min_length=1)
    client_secret: SecretStr = SecretStr("")
    token_endpoint_auth_method: MCPTokenEndpointAuthMethod

    @model_validator(mode="after")
    def validate_client_authentication(self):
        if self.token_endpoint_auth_method == "none":
            self.client_secret = SecretStr("")
        elif not self.client_secret.get_secret_value():
            raise ValueError(
                "Registered client authentication requires a client secret"
            )
        return self


def select_client_auth_method(
    metadata: dict[str, object]
) -> MCPTokenEndpointAuthMethod:
    supported = metadata.get(
        "token_endpoint_auth_methods_supported", ["client_secret_basic"]
    )
    if isinstance(supported, list):
        for method in ("none", "client_secret_basic", "client_secret_post"):
            if method in supported:
                return method
    raise ValueError(
        "This MCP server requires an unsupported client authentication method"
    )


def preregistered_client(server_url: str, secrets: Secrets) -> tuple[str, str] | None:
    """The configured ``(client_id, client_secret)`` for a server that needs a
    pre-registered OAuth app, ``("", "")`` when it needs one but none is
    configured, or ``None`` when the server is not one of them.

    Matches the URL's exact hostname, so a look-alike such as
    ``mcp.slack.com.evil.example`` never receives the platform's Slack app.
    """
    app = _preregistered_app(server_url)
    if not app:
        return None
    return getattr(secrets, app.client_id_field), getattr(
        secrets, app.client_secret_field
    )


def check_preregistered_endpoints(server_url: str, metadata: dict[str, object]):
    """Raise ``ValueError`` unless the discovered authorize and token endpoints
    are HTTPS URLs on the hosts the server's pre-registered app expects, so
    the platform's client secret is only ever sent to the provider itself."""
    app = _preregistered_app(server_url)
    if not app:
        return
    for key in ("authorization_endpoint", "token_endpoint"):
        if not _is_https_on(metadata.get(key), app.endpoint_hosts):
            raise ValueError(
                f"The {key} advertised by {urlsplit(server_url).hostname} is not "
                f"an HTTPS URL on {', '.join(sorted(app.endpoint_hosts))}"
            )


def preregistered_revocation_endpoint(
    server_url: str, revoke_url: str | None
) -> str | None:
    """The revocation endpoint to keep for ``server_url``. Revocation also
    authenticates with the client secret, so for a pre-registered app one that
    is not HTTPS on the provider's hosts is dropped; revocation is best-effort."""
    app = _preregistered_app(server_url)
    if not app or _is_https_on(revoke_url, app.endpoint_hosts):
        return revoke_url
    return None


def _preregistered_app(server_url: str) -> PreregisteredApp | None:
    try:
        parts = urlsplit(server_url)
        host = parts.hostname
    except ValueError:
        return None
    # Discovery decides where the secret is posted, so never over cleartext.
    if parts.scheme != "https":
        return None
    return PREREGISTERED_APPS.get(host or "")


def _is_https_on(url: object, hosts: frozenset[str]) -> bool:
    if not isinstance(url, str):
        return False
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return False
    return parts.scheme == "https" and host in hosts
