import json
from typing import Any, Type, TypeVar, get_origin


class ConversionError(Exception):
    pass


def __convert_list(value: Any) -> list:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    elif isinstance(value, dict):
        return list(value.items())
    elif isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return [value]
        else:
            return [value]
    else:
        return [value]


def __convert_dict(value: Any) -> dict:
    if isinstance(value, str):
        try:
            result = json.loads(value)
            if isinstance(result, dict):
                return result
            else:
                return {"value": result}
        except json.JSONDecodeError:
            return {"value": value}  # Fallback conversion
    elif isinstance(value, list):
        return {i: value[i] for i in range(len(value))}
    elif isinstance(value, tuple):
        return {i: value[i] for i in range(len(value))}
    elif isinstance(value, dict):
        return value
    else:
        return {"value": value}


def __convert_tuple(value: Any) -> tuple:
    if isinstance(value, (str, list, set)):
        return tuple(value)
    elif isinstance(value, dict):
        return tuple(value.items())
    elif isinstance(value, (int, float, bool)):
        return (value,)
    elif isinstance(value, tuple):
        return value
    else:
        return (value,)


def __convert_set(value: Any) -> set:
    if isinstance(value, (str, list, tuple)):
        return set(value)
    elif isinstance(value, dict):
        return set(value.items())
    elif isinstance(value, set):
        return value
    else:
        return {value}


def __convert_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    else:
        return json.dumps(value)


NUM = TypeVar("NUM", int, float)


def __convert_num(value: Any, num_type: Type[NUM]) -> NUM:
    if isinstance(value, (list, dict, tuple, set)):
        return num_type(len(value))
    elif isinstance(value, num_type):
        return value
    else:
        try:
            return num_type(float(value))
        except (ValueError, TypeError):
            return num_type(0)  # Fallback conversion


def __convert_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value.lower() in ["true", "1"]:
            return True
        else:
            return False
    else:
        return bool(value)


def convert(value: Any, target_type: Type):
    target_type = get_origin(target_type) or target_type
    if target_type not in [list, dict, tuple, str, set, int, float, bool]:
        return value
    if isinstance(value, target_type):
        return value
    if target_type is list:
        return __convert_list(value)
    elif target_type is dict:
        return __convert_dict(value)
    elif target_type is tuple:
        return __convert_tuple(value)
    elif target_type is str:
        return __convert_str(value)
    elif target_type is set:
        return __convert_set(value)
    elif target_type is int:
        return __convert_num(value, int)
    elif target_type is float:
        return __convert_num(value, float)
    elif target_type is bool:
        return __convert_bool(value)
    else:
        return value
