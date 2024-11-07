import json
from typing import Any, Type, TypeVar, overload

from fastapi.encoders import jsonable_encoder

from .type import convert


def to_dict(data) -> dict:
    return jsonable_encoder(data)


def dumps(data) -> str:
    return json.dumps(jsonable_encoder(data))


T = TypeVar("T")


@overload
def loads(data: str, *args, target_type: Type[T], **kwargs) -> T: ...


@overload
def loads(data: str, *args, **kwargs) -> Any: ...


def loads(data: str, *args, target_type: Type[T] | None = None, **kwargs) -> Any:
    parsed = json.loads(data, *args, **kwargs)
    return convert(parsed, target_type) if target_type else parsed
