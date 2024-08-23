from typing import Annotated, Any, Literal, Optional, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr, field_serializer


class _BaseCredentials(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    provider: str
    title: str

    @field_serializer("*")
    def dump_secret_strings(value: Any, _info):
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        return value


class OAuth2Credentials(_BaseCredentials):
    type: Literal["oauth2"] = "oauth2"
    access_token: SecretStr
    access_token_expires_at: int
    refresh_token: SecretStr
    refresh_token_expires_at: Optional[int]
    scopes: list[str]


class APIKeyCredentials(_BaseCredentials):
    type: Literal["api_key"] = "api_key"
    api_key: SecretStr
    expires_at: Optional[int]


Credentials = Annotated[
    OAuth2Credentials | APIKeyCredentials,
    Field(discriminator="type"),
]


class UserMetadata(BaseModel):
    integration_credentials: list[Credentials] = Field(default_factory=list)


class UserMetadataRaw(TypedDict, total=False):
    integration_credentials: list[dict]
