"""OpenAI Responses payloads for structured-output side calls."""

import json
import math
import time
import uuid
from fractions import Fraction
from typing import Any


class ResponseError(ValueError):
    pass


def response(body: dict[str, Any]) -> tuple[dict[str, Any], str]:
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ResponseError("model must be a non-empty string")

    text = body.get("text")
    response_format = text.get("format") if isinstance(text, dict) else None
    schema = (
        response_format.get("schema") if isinstance(response_format, dict) else None
    )
    if not isinstance(schema, dict):
        raise ResponseError("text.format.schema must be a JSON object")

    output_text = json.dumps(_schema_value(schema, schema), separators=(",", ":"))
    input_tokens = max(1, len(json.dumps(body)) // 4)
    output_tokens = max(1, len(output_text) // 4)
    response_id = f"resp_{uuid.uuid4().hex}"

    return (
        {
            "id": response_id,
            "created_at": int(time.time()),
            "model": model,
            "object": "response",
            "output": [
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "content": [
                        {
                            "annotations": [],
                            "text": output_text,
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "status": "completed",
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": output_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": input_tokens + output_tokens,
            },
        },
        output_text,
    )


def error(message_text: str) -> dict[str, Any]:
    return {
        "error": {
            "message": f"scripted-model: {message_text}",
            "type": "invalid_request_error",
            "param": None,
            "code": "scripted_model",
        }
    }


def _schema_value(schema: dict[str, Any], root: dict[str, Any], depth: int = 0) -> Any:
    if depth > 50:
        raise ResponseError("JSON schema is too deeply nested")

    reference = schema.get("$ref")
    if isinstance(reference, str):
        resolved = _resolve_reference(reference, root)
        schema = {
            **resolved,
            **{key: value for key, value in schema.items() if key != "$ref"},
        }

    if "const" in schema:
        return schema["const"]
    enum = schema.get("enum")
    if isinstance(enum, list):
        if not enum:
            raise ResponseError("JSON schema contains an empty enum")
        return enum[0]
    if "default" in schema:
        return schema["default"]

    for union in ("anyOf", "oneOf"):
        options = schema.get(union)
        if isinstance(options, list) and options:
            option = options[0]
            if not isinstance(option, dict):
                raise ResponseError(f"{union} entries must be JSON objects")
            return _schema_value(option, root, depth + 1)

    all_of = schema.get("allOf")
    if isinstance(all_of, list) and all_of:
        values = [
            _schema_value(option, root, depth + 1)
            for option in all_of
            if isinstance(option, dict)
        ]
        if values and all(isinstance(value, dict) for value in values):
            return {key: value for item in values for key, value in item.items()}
        if values:
            return values[0]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), "null")
    if schema_type is None:
        if "properties" in schema:
            schema_type = "object"
        elif "items" in schema:
            schema_type = "array"

    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ResponseError("object schema properties and required must be valid")
        result = {}
        for key in required:
            if not isinstance(key, str):
                raise ResponseError(f"required property {key!r} has no schema")
            property_schema = properties.get(key)
            if not isinstance(property_schema, dict):
                raise ResponseError(f"required property {key!r} has no schema")
            result[key] = _schema_value(property_schema, root, depth + 1)
        return result
    if schema_type == "array":
        items = schema.get("items", {})
        if not isinstance(items, dict):
            raise ResponseError("array items must be a JSON object")
        minimum = schema.get("minItems", 0)
        count = minimum if isinstance(minimum, int) and minimum > 0 else 0
        return [_schema_value(items, root, depth + 1) for _ in range(count)]
    if schema_type == "string":
        minimum = schema.get("minLength", 0)
        return "x" * minimum if isinstance(minimum, int) and minimum > 0 else ""
    if schema_type == "integer":
        return _bounded_number(schema, integer=True)
    if schema_type == "number":
        return _bounded_number(schema, integer=False)
    if schema_type == "boolean":
        return False
    if schema_type == "null" or schema_type is None:
        return None
    raise ResponseError(f"unsupported JSON schema type {schema_type!r}")


def _bounded_number(schema: dict[str, Any], *, integer: bool) -> int | float:
    minimum, minimum_exclusive = _lower_bound(schema)
    maximum, maximum_exclusive = _upper_bound(schema)
    multiple = _numeric_constraint(schema, "multipleOf")

    if multiple is not None and multiple <= 0:
        raise ResponseError("numeric schema multipleOf must be positive")

    if integer:
        if minimum is not None:
            minimum = Fraction(
                _floor_fraction(minimum) + 1
                if minimum_exclusive
                else _ceil_fraction(minimum)
            )
            minimum_exclusive = False
        if maximum is not None:
            maximum = Fraction(
                _ceil_fraction(maximum) - 1
                if maximum_exclusive
                else _floor_fraction(maximum)
            )
            maximum_exclusive = False
        if multiple is not None:
            multiple = Fraction(multiple.numerator)

    if (
        minimum is not None
        and maximum is not None
        and (
            minimum > maximum
            or (minimum == maximum and (minimum_exclusive or maximum_exclusive))
        )
    ):
        raise ResponseError("numeric schema has no value within its bounds")

    if multiple is not None:
        minimum_index = (
            _floor_fraction(minimum / multiple) + 1
            if minimum is not None and minimum_exclusive
            else _ceil_fraction(minimum / multiple) if minimum is not None else None
        )
        maximum_index = (
            _ceil_fraction(maximum / multiple) - 1
            if maximum is not None and maximum_exclusive
            else _floor_fraction(maximum / multiple) if maximum is not None else None
        )
        if (
            minimum_index is not None
            and maximum_index is not None
            and minimum_index > maximum_index
        ):
            raise ResponseError(
                "numeric schema has no value satisfying its constraints"
            )
        index = 0
        if minimum_index is not None and index < minimum_index:
            index = minimum_index
        if maximum_index is not None and index > maximum_index:
            index = maximum_index
        value = Fraction(index) * multiple
    else:
        value = _unscaled_value(
            minimum,
            minimum_exclusive,
            maximum,
            maximum_exclusive,
        )

    if not _within_bounds(
        value,
        minimum,
        minimum_exclusive,
        maximum,
        maximum_exclusive,
    ):
        raise ResponseError("numeric schema has no value satisfying its constraints")
    if integer or value.denominator == 1:
        return value.numerator
    try:
        return float(value)
    except OverflowError as error:
        raise ResponseError("numeric schema value is too large") from error


def _lower_bound(schema: dict[str, Any]) -> tuple[Fraction | None, bool]:
    minimum = _numeric_constraint(schema, "minimum")
    exclusive = _numeric_constraint(schema, "exclusiveMinimum")
    if exclusive is not None and (minimum is None or exclusive >= minimum):
        return exclusive, True
    return minimum, False


def _upper_bound(schema: dict[str, Any]) -> tuple[Fraction | None, bool]:
    maximum = _numeric_constraint(schema, "maximum")
    exclusive = _numeric_constraint(schema, "exclusiveMaximum")
    if exclusive is not None and (maximum is None or exclusive <= maximum):
        return exclusive, True
    return maximum, False


def _unscaled_value(
    minimum: Fraction | None,
    minimum_exclusive: bool,
    maximum: Fraction | None,
    maximum_exclusive: bool,
) -> Fraction:
    zero = Fraction(0)
    if _within_bounds(
        zero,
        minimum,
        minimum_exclusive,
        maximum,
        maximum_exclusive,
    ):
        return zero
    if minimum is not None and minimum >= zero:
        if not minimum_exclusive:
            return minimum
        return minimum + 1 if maximum is None else (minimum + maximum) / 2
    if maximum is None:
        raise ResponseError("numeric schema has no value within its bounds")
    if not maximum_exclusive:
        return maximum
    return maximum - 1 if minimum is None else (minimum + maximum) / 2


def _within_bounds(
    value: Fraction,
    minimum: Fraction | None,
    minimum_exclusive: bool,
    maximum: Fraction | None,
    maximum_exclusive: bool,
) -> bool:
    if minimum is not None and (
        value < minimum or (minimum_exclusive and value == minimum)
    ):
        return False
    return not (
        maximum is not None
        and (value > maximum or (maximum_exclusive and value == maximum))
    )


def _numeric_constraint(schema: dict[str, Any], key: str) -> Fraction | None:
    value = schema.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        raise ResponseError(f"numeric schema {key} must be finite")
    return Fraction(str(value))


def _ceil_fraction(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def _resolve_reference(reference: str, root: dict[str, Any]) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ResponseError(f"unsupported JSON schema reference {reference!r}")
    value: Any = root
    for part in reference[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or key not in value:
            raise ResponseError(f"unresolved JSON schema reference {reference!r}")
        value = value[key]
    if not isinstance(value, dict):
        raise ResponseError(f"JSON schema reference {reference!r} is not an object")
    return value
