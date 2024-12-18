from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    get_args,
)
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    SecretStr,
    field_serializer,
)
from pydantic_core import (
    CoreSchema,
    PydanticUndefined,
    PydanticUndefinedType,
    ValidationError,
    core_schema,
)

from backend.integrations.providers import ProviderName
from backend.util.settings import Secrets

if TYPE_CHECKING:
    from backend.data.block import BlockSchema

T = TypeVar("T")
logger = logging.getLogger(__name__)


class BlockSecret:
    def __init__(self, key: Optional[str] = None, value: Optional[str] = None):
        if value is not None:
            trimmed_value = value.strip()
            if value != trimmed_value:
                logger.debug(BlockSecret.TRIMMING_VALUE_MSG)
            self._value = trimmed_value
            return

        self._value = self.__get_secret(key)
        if self._value is None:
            raise ValueError(f"Secret {key} not found.")
        trimmed_value = self._value.strip()
        if self._value != trimmed_value:
            logger.debug(BlockSecret.TRIMMING_VALUE_MSG)
        self._value = trimmed_value

    TRIMMING_VALUE_MSG: ClassVar[str] = "Provided secret value got trimmed."
    STR: ClassVar[str] = "<secret>"
    SECRETS: ClassVar[Secrets] = Secrets()

    def __repr__(self):
        return BlockSecret.STR

    def __str__(self):
        return BlockSecret.STR

    @staticmethod
    def __get_secret(key: str | None):
        if not key or not hasattr(BlockSecret.SECRETS, key):
            return None
        return getattr(BlockSecret.SECRETS, key)

    def get_secret_value(self):
        trimmed_value = str(self._value).strip()
        if self._value != trimmed_value:
            logger.info(BlockSecret.TRIMMING_VALUE_MSG)
        return trimmed_value

    @classmethod
    def parse_value(cls, value: Any) -> BlockSecret:
        if isinstance(value, BlockSecret):
            return value
        return BlockSecret(value=value)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> dict[str, Any]:
        return {
            "type": "string",
        }

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        validate_fun = core_schema.no_info_plain_validator_function(cls.parse_value)
        return core_schema.json_or_python_schema(
            json_schema=validate_fun,
            python_schema=validate_fun,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda val: BlockSecret.STR
            ),
        )


def SecretField(
    value: Optional[str] = None,
    key: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    placeholder: Optional[str] = None,
    **kwargs,
) -> BlockSecret:
    return SchemaField(
        BlockSecret(key=key, value=value),
        title=title,
        description=description,
        placeholder=placeholder,
        secret=True,
        **kwargs,
    )


def SchemaField(
    default: T | PydanticUndefinedType = PydanticUndefined,
    *args,
    default_factory: Optional[Callable[[], T]] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    placeholder: Optional[str] = None,
    advanced: Optional[bool] = False,
    secret: bool = False,
    exclude: bool = False,
    hidden: Optional[bool] = None,
    depends_on: list[str] | None = None,
    **kwargs,
) -> T:
    json_extra = {
        k: v
        for k, v in {
            "placeholder": placeholder,
            "secret": secret,
            "advanced": advanced,
            "hidden": hidden,
            "depends_on": depends_on,
        }.items()
        if v is not None
    }

    return Field(
        default,
        *args,
        default_factory=default_factory,
        title=title,
        description=description,
        exclude=exclude,
        json_schema_extra=json_extra,
        **kwargs,
    )  # type: ignore


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
    """Unix timestamp (seconds) indicating when this OAuth state expires"""
    scopes: list[str]


class UserMetadata(BaseModel):
    integration_credentials: list[Credentials] = Field(default_factory=list)
    integration_oauth_states: list[OAuthState] = Field(default_factory=list)


class UserMetadataRaw(TypedDict, total=False):
    integration_credentials: list[dict]
    integration_oauth_states: list[dict]


class UserIntegrations(BaseModel):
    credentials: list[Credentials] = Field(default_factory=list)
    oauth_states: list[OAuthState] = Field(default_factory=list)


CP = TypeVar("CP", bound=ProviderName)
CT = TypeVar("CT", bound=CredentialsType)


CREDENTIALS_FIELD_NAME = "credentials"


class CredentialsMetaInput(BaseModel, Generic[CP, CT]):
    id: str
    title: Optional[str] = None
    provider: CP
    type: CT

    @staticmethod
    def _add_json_schema_extra(schema, cls: CredentialsMetaInput):
        schema["credentials_provider"] = get_args(
            cls.model_fields["provider"].annotation
        )
        schema["credentials_types"] = get_args(cls.model_fields["type"].annotation)

    model_config = ConfigDict(
        json_schema_extra=_add_json_schema_extra,  # type: ignore
    )

    @classmethod
    def validate_credentials_field_schema(cls, model: type["BlockSchema"]):
        """Validates the schema of a `credentials` field"""
        field_schema = model.jsonschema()["properties"][CREDENTIALS_FIELD_NAME]
        try:
            schema_extra = _CredentialsFieldSchemaExtra[CP, CT].model_validate(
                field_schema
            )
        except ValidationError as e:
            if "Field required [type=missing" not in str(e):
                raise

            raise TypeError(
                "Field 'credentials' JSON schema lacks required extra items: "
                f"{field_schema}"
            ) from e

        if (
            len(schema_extra.credentials_provider) > 1
            and not schema_extra.discriminator
        ):
            raise TypeError("Multi-provider CredentialsField requires discriminator!")


class _CredentialsFieldSchemaExtra(BaseModel, Generic[CP, CT]):
    # TODO: move discrimination mechanism out of CredentialsField (frontend + backend)
    credentials_provider: list[CP]
    credentials_scopes: Optional[list[str]] = None
    credentials_types: list[CT]
    discriminator: Optional[str] = None
    discriminator_mapping: Optional[dict[str, CP]] = None


def CredentialsField(
    required_scopes: set[str] = set(),
    *,
    discriminator: Optional[str] = None,
    discriminator_mapping: Optional[dict[str, Any]] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs,
) -> CredentialsMetaInput:
    """
    `CredentialsField` must and can only be used on fields named `credentials`.
    This is enforced by the `BlockSchema` base class.
    """

    field_schema_extra = {
        k: v
        for k, v in {
            "credentials_scopes": list(required_scopes) or None,
            "discriminator": discriminator,
            "discriminator_mapping": discriminator_mapping,
        }.items()
        if v is not None
    }

    return Field(
        title=title,
        description=description,
        json_schema_extra=field_schema_extra,  # validated on BlockSchema init
        **kwargs,
    )


class ContributorDetails(BaseModel):
    name: str = Field(title="Name", description="The name of the contributor.")
