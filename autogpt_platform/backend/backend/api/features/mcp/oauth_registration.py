from pydantic import BaseModel, Field, SecretStr, model_validator

from backend.blocks.mcp.oauth import MCPTokenEndpointAuthMethod
from backend.util.settings import Secrets

# MCP servers that refuse Dynamic Client Registration and only accept an OAuth
# app registered with them in advance, keyed by server host. Each maps to the
# ``Secrets`` fields holding that app's client id and secret.
PREREGISTERED_CLIENT_SECRETS: dict[str, tuple[str, str]] = {
    "mcp.slack.com": ("slack_mcp_client_id", "slack_mcp_client_secret"),
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


def preregistered_client(host: str, secrets: Secrets) -> tuple[str, str] | None:
    """The configured ``(client_id, client_secret)`` for a server that needs a
    pre-registered OAuth app, ``("", "")`` when it needs one but none is
    configured, or ``None`` when the server is not one of them."""
    fields = PREREGISTERED_CLIENT_SECRETS.get(host)
    if not fields:
        return None
    id_field, secret_field = fields
    return getattr(secrets, id_field), getattr(secrets, secret_field)
