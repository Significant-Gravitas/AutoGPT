from pydantic import BaseModel, Field, SecretStr, model_validator

from backend.blocks.mcp.oauth import MCPTokenEndpointAuthMethod


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
