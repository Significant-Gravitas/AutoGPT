"""Block inputs and outputs are arbitrary JSON: a number, a boolean, a list, an
object or null as often as a string. Every external API field that carries them
must accept any JSON value. A field typed narrower fails on the first run that
produces another type: with a 500 on a response, as v1's run results did on a
number (Sentry AUTOGPT-SERVER-AG3), or with a 422 on a request.

Checked against the OpenAPI document each version publishes, the way
`v2/contract_shape_test.py` checks v2's other rules.
"""

from typing import Any

import pytest

from backend.api.external.v1.app import v1_app
from backend.api.external.v2.app import v2_app

# Fields with these names carry block inputs or outputs, inside any lists and
# string-keyed maps.
BLOCK_VALUE_FIELDS = {
    "input",
    "inputs",
    "output",
    "outputs",
    "input_data",
    "output_data",
    "input_default",
    "node_input",
}

# Fields with one of those names that carry something else: "Model.field" -> why.
NOT_BLOCK_VALUES: dict[str, str] = {}

# Keywords that restrict a schema. A schema with none of them, such as the
# `{"title": ...}` that `Any` publishes, accepts any value.
RESTRICTING_KEYWORDS = {
    "$ref",
    "type",
    "anyOf",
    "oneOf",
    "allOf",
    "enum",
    "const",
    "not",
}

# Each version's spec, and a field the walk must reach there.
SPECS = {
    "v1": (v1_app, "GraphExecutionResult.output"),
    "v2": (v2_app, "AgentGraphRunDetails.outputs"),
}


@pytest.mark.parametrize("version", SPECS)
def test_every_field_carrying_block_values_accepts_any_json(version: str) -> None:
    app, reached = SPECS[version]
    spec = app.openapi()
    fields = {
        f"{model}.{field}": field_schema
        for model, schema in spec["components"]["schemas"].items()
        for field, field_schema in schema.get("properties", {}).items()
        if field in BLOCK_VALUE_FIELDS and f"{model}.{field}" not in NOT_BLOCK_VALUES
    }
    narrowed = {
        name: where
        for name, field_schema in fields.items()
        if (where := _narrowed(spec, field_schema, name))
    }

    assert reached in fields, f"the walk no longer reaches {reached}"
    assert not narrowed, (
        f"These {version} fields carry block inputs or outputs, which can be any "
        f"JSON value, but accept less: {narrowed}. Type the values Any or "
        "JsonValue, e.g. dict[str, Any] or list[Any]. If a field holds "
        "something else, add it to NOT_BLOCK_VALUES and say why."
    )


def _narrowed(spec: dict[str, Any], schema: Any, at: str) -> list[str]:
    """Where a value under `schema` must be narrower than any JSON value."""
    if schema is True:
        return []
    if schema is False:
        return [f"{at} accepts no value"]
    if not schema.keys() & RESTRICTING_KEYWORDS:
        return []
    if "$ref" in schema:
        return _narrowed(spec, _resolve(spec, schema["$ref"]), at)
    if branches := schema.get("anyOf") or schema.get("oneOf"):
        # Optional[...] adds a null branch, which narrows nothing a value holds.
        found = [_narrowed(spec, b, at) for b in branches if b != {"type": "null"}]
        return [] if [] in found else [where for f in found for where in f]
    if parts := schema.get("allOf"):
        return [where for part in parts for where in _narrowed(spec, part, at)]
    if schema.get("type") == "array":
        return _narrowed(spec, schema.get("items", True), f"{at}[]")
    if schema.get("type") == "object" and "properties" not in schema:
        return _narrowed(spec, schema.get("additionalProperties", True), f"{at}{{}}")
    if "properties" in schema:
        return [f"{at} is a model with fixed fields"]
    return [f"{at} is {schema.get('type') or schema}"]


def _resolve(spec: dict[str, Any], ref: str) -> Any:
    node = spec
    for part in ref.removeprefix("#/").split("/"):
        node = node[part]
    return node
