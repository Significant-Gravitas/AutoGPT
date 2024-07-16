from typing import ClassVar, Optional

from pydantic import Field
from pydantic.fields import FieldInfo

from autogpt_server.util.settings import Secrets


class BlockSecret:
    def __init__(self, key=None, value=None):
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


def SecretField(
    value: Optional[BlockSecret] = None,
    key: Optional[str] = None,
    description: str = "",
    **kwargs,
) -> BlockSecret:
    field_info: FieldInfo = Field(
        BlockSecret(key=key, value=value),
        description=description,
        exclude=True,
        **kwargs,
    )

    return field_info  # type: ignore
