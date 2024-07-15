from typing import Any, Callable, Optional, TypeVar

from pydantic import Field, SecretStr
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined, PydanticUndefinedType

from autogpt_server.util.settings import Secrets


SECRETS = Secrets()


def SecretField(
    value: Optional[SecretStr | str] = None,
    key: Optional[str] = None,
    description: str = "",
    **kwargs,
) -> SecretStr:
    if not value and key and hasattr(SECRETS, key):
        value = getattr(SECRETS, key)

    if value is None:
        raise ValueError(f"Secret {key} not found.")
    elif isinstance(value, str):
        value = SecretStr(value)
    
    field_info: FieldInfo = Field(
        value,
        description=description,
        **kwargs,
    )

    return field_info  # type: ignore
