from __future__ import annotations

import logging
from typing import Any, Callable, ClassVar, Generic, Optional, TypeVar

from autogpt_libs.supabase_integration_credentials_store.types import CredentialsType
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import (
    CoreSchema,
    PydanticUndefined,
    PydanticUndefinedType,
    core_schema,
)

from backend.util.settings import Secrets

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
    advanced: Optional[bool] = None,
    secret: bool = False,
    exclude: bool = False,
    **kwargs,
) -> T:
    json_extra = {
        k: v
        for k, v in {
            "placeholder": placeholder,
            "secret": secret,
            "advanced": advanced,
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
    )


CP = TypeVar("CP", bound=str)
CT = TypeVar("CT", bound=CredentialsType)


CREDENTIALS_FIELD_NAME = "credentials"


class CredentialsMetaInput(BaseModel, Generic[CP, CT]):
    id: str
    title: Optional[str] = None
    provider: CP
    type: CT


class CredentialsFieldSchemaExtra(BaseModel, Generic[CP, CT]):
    # TODO: move discrimination mechanism out of CredentialsField (frontend + backend)
    credentials_provider: list[CP]
    credentials_scopes: Optional[list[str]]
    credentials_types: list[CT]
    discriminator: Optional[str] = None
    discriminator_mapping: Optional[dict[str, CP]] = None


def CredentialsField(
    provider: CP | list[CP],
    supported_credential_types: set[CT],
    required_scopes: set[str] = set(),
    *,
    discriminator: Optional[str] = None,
    discriminator_mapping: Optional[dict[str, Any]] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs,
) -> CredentialsMetaInput[CP, CT]:
    """
    `CredentialsField` must and can only be used on fields named `credentials`.
    This is enforced by the `BlockSchema` base class.
    """
    if not isinstance(provider, str) and len(provider) > 1 and not discriminator:
        raise TypeError("Multi-provider CredentialsField requires discriminator!")

    field_schema_extra = CredentialsFieldSchemaExtra[CP, CT](
        credentials_provider=[provider] if isinstance(provider, str) else provider,
        credentials_scopes=list(required_scopes) or None,  # omit if empty
        credentials_types=list(supported_credential_types),
        discriminator=discriminator,
        discriminator_mapping=discriminator_mapping,
    )

    return Field(
        title=title,
        description=description,
        json_schema_extra=field_schema_extra.model_dump(exclude_none=True),
        **kwargs,
    )


class ContributorDetails(BaseModel):
    name: str = Field(title="Name", description="The name of the contributor.")
