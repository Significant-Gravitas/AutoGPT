from __future__ import annotations
from typing import Any, ClassVar, Optional

from pydantic import Field, GetCoreSchemaHandler
from pydantic.fields import FieldInfo
from pydantic_core import CoreSchema, core_schema

from autogpt_server.util.settings import Secrets


class BlockSecret:
    def __init__(self, key: Optional[str] = None, value: Optional[str] = None):
        if value is not None:
            self._value = value
            return

        self._value = self.__get_secret(key)
        if self._value is None:
            raise ValueError(f"Secret {key} not found.")

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
        return str(self._value)
    
    @classmethod
    def parse_value(cls, value: Any) -> BlockSecret:
        if isinstance(value, BlockSecret):
            return value
        return BlockSecret(value=value)
    
    @classmethod
    def __get_pydantic_json_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler) -> dict[str, Any]:
        return {
            "type": "string",
            "title": "BlockSecret",
        }

    @classmethod
    def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
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
    **kwargs,
) -> BlockSecret:
    field_info: FieldInfo = Field(
        BlockSecret(key=key, value=value),
        title=title,
        description=description,
        **kwargs,
    )

    return field_info  # type: ignore
