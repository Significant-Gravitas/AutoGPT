from typing import Annotated, Any, Literal, Optional, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr, field_serializer


class _BaseCredentials(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    provider: str
    title: Optional[str]

    @field_serializer("*")
    def dump_secret_strings(value: Any, _info):
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        return value


class OAuth2Credentials(_BaseCredentials):
    type: Literal["oauth2"] = "oauth2"
    username: Optional[str]
    """Username of the third-party service user that these credentials belong to"""
    access_token: SecretStr
    access_token_expires_at: Optional[int]
    """Unix timestamp (seconds) indicating when the access token expires (if at all)"""
    refresh_token: Optional[SecretStr]
    refresh_token_expires_at: Optional[int]
    """Unix timestamp (seconds) indicating when the refresh token expires (if at all)"""
    scopes: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def bearer(self) -> str:
        return f"Bearer {self.access_token.get_secret_value()}"


class APIKeyCredentials(_BaseCredentials):
    type: Literal["api_key"] = "api_key"
    api_key: SecretStr
    expires_at: Optional[int]
    """Unix timestamp (seconds) indicating when the API key expires (if at all)"""

    def bearer(self) -> str:
        return f"Bearer {self.api_key.get_secret_value()}"


Credentials = Annotated[
    OAuth2Credentials | APIKeyCredentials,
    Field(discriminator="type"),
]


CredentialsType = Literal["api_key", "oauth2"]


class OAuthState(BaseModel):
    token: str
    provider: str
    expires_at: int
    code_verifier: Optional[str] = None
    scopes: list[str]
    """Unix timestamp (seconds) indicating when this OAuth state expires"""


class UserMetadata(BaseModel):
    integration_credentials: list[Credentials] = Field(default_factory=list)
    integration_oauth_states: list[OAuthState] = Field(default_factory=list)


class UserMetadataRaw(TypedDict, total=False):
    integration_credentials: list[dict]
    integration_oauth_states: list[dict]


class UserIntegrations(BaseModel):
    credentials: list[Credentials] = Field(default_factory=list)
    oauth_states: list[OAuthState] = Field(default_factory=list)
