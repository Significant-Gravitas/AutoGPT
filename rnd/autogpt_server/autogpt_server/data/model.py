from typing import Any, Callable, Optional, TypeVar

from pydantic import Field
from pydantic_core import PydanticUndefined, PydanticUndefinedType


T = TypeVar("T")


def SchemaField(
    default: T | PydanticUndefinedType = PydanticUndefined,
    *args,
    default_factory: Optional[Callable[[], T]] = None,
    title: str = "",
    description: str = "",
    placeholder: str = "",
    exclude: bool = False,
    **kwargs,
) -> T:
    json_extra: dict[str, Any] = {}
    if placeholder:
        json_extra["placeholder"] = placeholder

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
