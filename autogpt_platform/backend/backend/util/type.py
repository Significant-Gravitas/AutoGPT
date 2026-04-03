import json
import types
from typing import Any, Type, TypeVar, Union, cast, get_args, get_origin, overload

from prisma import Json as PrismaJson


def _is_type_or_subclass(origin: Any, target_type: type) -> bool:
    """Check if origin is exactly the target type or a subclass of it."""
    return origin is target_type or (
        isinstance(origin, type) and issubclass(origin, target_type)
    )


class ConversionError(ValueError):
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


def _try_convert(value: Any, target_type: Any, raise_on_mismatch: bool) -> Any:
    origin = get_origin(target_type)
    args = get_args(target_type)

    # Handle Union types (including Optional which is Union[T, None])
    if origin is Union or origin is types.UnionType:
        # Handle None values for Optional types
        if value is None:
            if type(None) in args:
                return None
            elif raise_on_mismatch:
                raise TypeError(f"Value {value} is not of expected type {target_type}")
            else:
                return value

        # Try to convert to each type in the union, excluding None
        non_none_types = [arg for arg in args if arg is not type(None)]

        # Try each type in the union, using the original raise_on_mismatch behavior
        for arg_type in non_none_types:
            try:
                return _try_convert(value, arg_type, raise_on_mismatch)
            except (TypeError, ValueError, ConversionError):
                continue

        # If no conversion succeeded
        if raise_on_mismatch:
            raise TypeError(f"Value {value} is not of expected type {target_type}")
        else:
            return value

    if origin is None:
        origin = target_type
    # Early return for unsupported types (skip subclasses of supported types)
    supported_types = [list, dict, tuple, str, set, int, float, bool]
    if origin not in supported_types and not (
        isinstance(origin, type) and any(issubclass(origin, t) for t in supported_types)
    ):
        return value

    # Handle the case when value is already of the target type
    if isinstance(value, origin):
        if not args:
            return value
        else:
            # Need to convert elements
            if origin is list:
                return [convert(v, args[0]) for v in value]
            elif origin is tuple:
                # Tuples can have multiple types
                if len(args) == 1:
                    return tuple(convert(v, args[0]) for v in value)
                else:
                    return tuple(convert(v, t) for v, t in zip(value, args))
            elif origin is dict:
                key_type, val_type = args
                return {
                    convert(k, key_type): convert(v, val_type) for k, v in value.items()
                }
            elif origin is set:
                return {convert(v, args[0]) for v in value}
            else:
                return value
    elif raise_on_mismatch:
        raise TypeError(f"Value {value} is not of expected type {target_type}")
    else:
        # Need to convert value to the origin type
        if _is_type_or_subclass(origin, list):
            converted_list = __convert_list(value)
            if args:
                converted_list = [convert(v, args[0]) for v in converted_list]
            return origin(converted_list) if origin is not list else converted_list
        elif _is_type_or_subclass(origin, dict):
            converted_dict = __convert_dict(value)
            if args:
                key_type, val_type = args
                converted_dict = {
                    convert(k, key_type): convert(v, val_type)
                    for k, v in converted_dict.items()
                }
            return origin(converted_dict) if origin is not dict else converted_dict
        elif _is_type_or_subclass(origin, tuple):
            converted_tuple = __convert_tuple(value)
            if args:
                if len(args) == 1:
                    converted_tuple = tuple(
                        convert(v, args[0]) for v in converted_tuple
                    )
                else:
                    converted_tuple = tuple(
                        convert(v, t) for v, t in zip(converted_tuple, args)
                    )
            return origin(converted_tuple) if origin is not tuple else converted_tuple
        elif _is_type_or_subclass(origin, str):
            converted_str = __convert_str(value)
            return origin(converted_str) if origin is not str else converted_str
        elif _is_type_or_subclass(origin, set):
            value = __convert_set(value)
            if args:
                return {convert(v, args[0]) for v in value}
            else:
                return value
        elif _is_type_or_subclass(origin, bool):
            return __convert_bool(value)
        elif _is_type_or_subclass(origin, int):
            return __convert_num(value, int)
        elif _is_type_or_subclass(origin, float):
            return __convert_num(value, float)
        else:
            return value


T = TypeVar("T")
TT = TypeVar("TT")


def type_match(value: Any, target_type: Type[T]) -> T:
    return cast(T, _try_convert(value, target_type, raise_on_mismatch=True))


@overload
def convert(value: Any, target_type: Type[T]) -> T: ...


@overload
def convert(value: Any, target_type: Any) -> Any: ...


def convert(value: Any, target_type: Any) -> Any:
    try:
        if isinstance(value, PrismaJson):
            value = value.data
        return _try_convert(value, target_type, raise_on_mismatch=False)
    except Exception as e:
        raise ConversionError(f"Failed to convert {value} to {target_type}") from e


def _value_satisfies_type(value: Any, target: Any) -> bool:
    """Check whether *value* already satisfies *target*, including inner elements.

    For union types this checks each member; for generic container types it
    recursively checks that inner elements satisfy the type args (e.g. every
    element in a ``list[str]`` is a ``str``).  Returns ``False`` when uncertain
    so the caller falls through to :func:`convert`.
    """
    # typing.Any cannot be used with isinstance(); treat as always satisfied.
    if target is Any:
        return True

    origin = get_origin(target)

    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(target) if a is not type(None)]
        return any(_value_satisfies_type(value, member) for member in non_none)

    # Generic container type (e.g. list[str], dict[str, int])
    if origin is not None:
        # Guard: origin may not be a runtime type (e.g. Literal)
        if not isinstance(origin, type):
            return False
        if not isinstance(value, origin):
            return False
        args = get_args(target)
        if not args:
            return True
        # Check inner elements satisfy the type args
        if _is_type_or_subclass(origin, list):
            return all(_value_satisfies_type(v, args[0]) for v in value)
        if _is_type_or_subclass(origin, dict) and len(args) >= 2:
            return all(
                _value_satisfies_type(k, args[0]) and _value_satisfies_type(v, args[1])
                for k, v in value.items()
            )
        if (
            _is_type_or_subclass(origin, set) or _is_type_or_subclass(origin, frozenset)
        ) and args:
            return all(_value_satisfies_type(v, args[0]) for v in value)
        if _is_type_or_subclass(origin, tuple):
            # Homogeneous tuple[T, ...] — single type + Ellipsis
            if len(args) == 2 and args[1] is Ellipsis:
                return all(_value_satisfies_type(v, args[0]) for v in value)
            # Heterogeneous tuple[T1, T2, ...] — positional types
            if len(value) != len(args):
                return False
            return all(_value_satisfies_type(v, t) for v, t in zip(value, args))
        # Unhandled generic origin — fall through to convert()
        return False

    # Simple type (e.g. str, int)
    if isinstance(target, type):
        try:
            return isinstance(value, target)
        except TypeError:
            # TypedDict and some typing constructs don't support isinstance checks.
            # For TypedDict, check if value is a dict with the required keys.
            if isinstance(value, dict) and hasattr(target, "__required_keys__"):
                return all(k in value for k in target.__required_keys__)
            return False

    return False


def coerce_inputs_to_schema(data: dict[str, Any], schema: type) -> None:
    """Coerce *data* values in-place to match *schema*'s field types.

    Uses ``model_fields`` (not ``__annotations__``) so inherited fields are
    included.  Skips coercion when the value already satisfies the target
    type — in particular for union-typed fields where the value matches one
    member but differs from the annotation object itself.

    This is the single authoritative coercion step shared by the executor
    (``validate_exec``) and the CoPilot (``execute_block``).
    """
    for name, field_info in schema.model_fields.items():
        value = data.get(name)
        if value is None:
            continue
        target = field_info.annotation
        if target is None:
            continue
        if _value_satisfies_type(value, target):
            continue
        data[name] = convert(value, target)


class FormattedStringType(str):
    string_format: str

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        _ = source_type  # unused parameter required by pydantic
        return handler(str)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        json_schema = handler(core_schema)
        json_schema["format"] = cls.string_format
        return json_schema


class MediaFileType(FormattedStringType):
    """
    MediaFile is a string that represents a file. It can be one of the following:
        - Data URI: base64 encoded media file. See https://developer.mozilla.org/en-US/docs/Web/URI/Schemes/data/
        - URL: Media file hosted on the internet, it starts with http:// or https://.
        - Local path (anything else): A temporary file path living within graph execution time.

    Note: Replace this type alias into a proper class, when more information is needed.
    """

    string_format = "file"


class LongTextType(FormattedStringType):
    string_format = "long-text"


class ShortTextType(FormattedStringType):
    string_format = "short-text"
