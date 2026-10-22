"""OpenAI Responses payloads for structured-output side calls."""

import json
import math
import time
import uuid
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
        minimum = schema.get("minimum", 0)
        if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
            return max(0, math.ceil(minimum))
        return 0
    if schema_type == "number":
        minimum = schema.get("minimum", 0)
        if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
            return max(0.0, float(minimum))
        return 0.0
    if schema_type == "boolean":
        return False
    if schema_type == "null" or schema_type is None:
        return None
    raise ResponseError(f"unsupported JSON schema type {schema_type!r}")


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
