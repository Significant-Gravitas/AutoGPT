"""Block inputs and outputs are arbitrary JSON: a number, a boolean, a list, an
object or null as often as a string. Every external API field that carries them
must accept any JSON value. A field typed narrower fails on the first run that
produces another type: with a 500 on a response, as v1's run results did on a
number (Sentry AUTOGPT-SERVER-AG3), or with a 422 on a request.

Checked against the OpenAPI document each version publishes, the way
`v2/contract_shape_test.py` checks v2's other rules.
"""

from typing import Annotated, Any

import pytest
from fastapi import FastAPI
from pydantic import Field, JsonValue, create_model

from backend.api.external.v1.app import v1_app
from backend.api.external.v2.app import v2_app

# A field with one of these names carries block inputs or outputs.
BLOCK_VALUE_NAMES = {
    "input",
    "inputs",
    "output",
    "outputs",
    "input_data",
    "output_data",
    "input_default",
    "node_input",
}

# Each such field, with the containers the API holds its values in, outermost
# first: "list" for an array, "map" for an object keyed by name. Where they end,
# the schema must accept any JSON value, a list or a map included.
BLOCK_VALUE_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "v1": {
        "Body_execute_graph_graphs__graph_id__execute__graph_version__post.node_input": (
            "map",
        ),
        "ExecutionNode.input": (),
        "ExecutionNode.output": ("map", "list"),
        "GraphExecutionResult.output": ("list", "map"),
        "Node.input_default": ("map",),
        "NodeModel.input_default": ("map",),
        "RunAgentRequest.inputs": ("map",),
    },
    "v2": {
        "AgentGraphRun.inputs": ("map",),
        "AgentGraphRunDetails.inputs": ("map",),
        "AgentGraphRunDetails.outputs": ("map", "list"),
        "AgentNodeExecution.inputs": ("map",),
        "AgentNodeExecution.outputs": ("map", "list"),
        "AgentPreset.inputs": ("map",),
        "AgentPresetCreateRequest.inputs": ("map",),
        "AgentPresetRunRequest.inputs": ("map",),
        "AgentPresetUpdateRequest.inputs": ("map",),
        "AgentRunRequest.inputs": ("map",),
        "AgentRunSchedule.inputs": ("map",),
        "AgentRunScheduleCreateRequest.inputs": ("map",),
        "AgentTriggerSetupRequest.inputs": ("map",),
        "GraphNode.input_default": ("map",),
    },
}

# Fields with one of those names that carry something else: "Model.field" -> why.
NOT_BLOCK_VALUES = {
    "ValidationError.input": "FastAPI's 422 detail, echoing a rejected request value",
}

# Keywords that only annotate a schema. A schema with no other keyword, like `{}`
# or the `{"title": ...}` that `Any` publishes, accepts any JSON value. Any other
# keyword can narrow what it accepts, so the walk treats it as narrowing.
ANNOTATION_KEYWORDS = {
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$comment",
}

# Each container: its JSON type, the keyword holding its values' schema, and how
# a failure message marks a step into it.
CONTAINERS = {
    "list": ("array", "items", "[]"),
    "map": ("object", "additionalProperties", "{}"),
}

SPECS = {"v1": v1_app, "v2": v2_app}


@pytest.mark.parametrize("version", SPECS)
def test_every_field_carrying_block_values_accepts_any_json(version: str) -> None:
    spec = SPECS[version].openapi()
    named = {
        f"{model}.{field}": field_schema
        for model, schema in spec["components"]["schemas"].items()
        for field, field_schema in schema.get("properties", {}).items()
        if field in BLOCK_VALUE_NAMES and f"{model}.{field}" not in NOT_BLOCK_VALUES
    }
    envelopes = BLOCK_VALUE_FIELDS[version]

    assert named.keys() == envelopes.keys(), (
        f"List each {version} field named for block values in BLOCK_VALUE_FIELDS, "
        "with the containers that hold its values, or in NOT_BLOCK_VALUES with why. "
        f"Not listed: {sorted(named.keys() - envelopes.keys())}. "
        f"No longer published: {sorted(envelopes.keys() - named.keys())}."
    )
    narrowed = {
        name: where
        for name, envelope in envelopes.items()
        if (where := _narrowed(spec, named[name], envelope, name))
    }
    assert not narrowed, (
        f"These {version} fields carry block inputs or outputs, which can be any "
        f"JSON value, but accept less: {narrowed}. Inside the containers "
        "BLOCK_VALUE_FIELDS lists, type the values Any or JsonValue, e.g. "
        "dict[str, Any] for ('map',)."
    )


@pytest.mark.parametrize(
    "annotation, narrowed",
    [
        pytest.param(list[dict[str, Any]] | None, [], id="any"),
        pytest.param(list[dict[str, JsonValue]], [], id="json-value"),
        pytest.param(list[dict[str, str]], ["Probe.field[]{} is string"], id="str"),
        pytest.param(
            list[dict[str, dict[str, Any]]],
            ["Probe.field[]{} is a map"],
            id="extra-map",
        ),
        pytest.param(
            list[dict[str, list[Any]]], ["Probe.field[]{} is a list"], id="extra-list"
        ),
        pytest.param(
            dict[str, list[Any]],
            ["Probe.field is a map, not a list"],
            id="wrong-container",
        ),
        pytest.param(
            Annotated[list[dict[str, Any]], Field(min_length=1)],
            ["Probe.field is a list restricted by ['minItems']"],
            id="restricted-container",
        ),
    ],
)
def test_the_walk_checks_the_values_where_their_containers_end(
    annotation: Any, narrowed: list[str]
) -> None:
    """For a list of maps of block values, the shape of v1's run results."""
    app = FastAPI()
    app.get("/probe", response_model=create_model("Probe", field=(annotation, ...)))(
        _no_content
    )
    spec = app.openapi()
    field = spec["components"]["schemas"]["Probe"]["properties"]["field"]

    assert _narrowed(spec, field, ("list", "map"), "Probe.field") == narrowed


@pytest.mark.parametrize(
    "schema, narrowed",
    [
        pytest.param({}, [], id="empty"),
        pytest.param(True, [], id="true"),
        pytest.param({"description": "x"}, [], id="annotation-only"),
        *(
            pytest.param(
                {keyword: value}, [f"value is restricted by ['{keyword}']"], id=keyword
            )
            for keyword, value in [
                ("maxLength", 1),
                ("pattern", "x"),
                ("minimum", 0),
                ("enum", [1]),
                ("const", 1),
                ("not", {}),
                ("minItems", 1),
            ]
        ),
    ],
)
def test_a_value_schema_accepts_any_json_only_with_annotations_alone(
    schema: Any, narrowed: list[str]
) -> None:
    assert _narrowed({}, schema, (), "value") == narrowed


def _narrowed(
    spec: dict[str, Any], schema: Any, envelope: tuple[str, ...], at: str
) -> list[str]:
    """Where a value inside `envelope` must be narrower than any JSON value."""
    if schema is True:
        return []
    if schema is False:
        return [f"{at} accepts no value"]
    keywords = schema.keys() - ANNOTATION_KEYWORDS
    if not keywords:
        return []
    if keywords == {"$ref"}:
        return _narrowed(spec, _resolve(spec, schema["$ref"]), envelope, at)
    if keywords in ({"anyOf"}, {"oneOf"}):
        branches = schema.get("anyOf") or schema.get("oneOf")
        # Optional[...] adds a null branch, which narrows nothing a value holds.
        found = [
            _narrowed(spec, b, envelope, at) for b in branches if b != {"type": "null"}
        ]
        return [] if [] in found else [where for f in found for where in f]
    if keywords == {"allOf"}:
        parts = schema["allOf"]
        return [w for part in parts for w in _narrowed(spec, part, envelope, at)]
    if not envelope:
        return [f"{at} is {_kind(schema, keywords)}"]
    return _container_narrowed(spec, schema, keywords, envelope, at)


def _container_narrowed(
    spec: dict[str, Any],
    schema: dict[str, Any],
    keywords: set[str],
    envelope: tuple[str, ...],
    at: str,
) -> list[str]:
    """Where the outermost container in `envelope`, or what it holds, narrows."""
    container, *inner = envelope
    json_type, values, step = CONTAINERS[container]
    if schema.get("type") != json_type or "properties" in schema:
        return [f"{at} is {_kind(schema, keywords)}, not a {container}"]
    if extra := sorted(keywords - {"type", values}):
        return [f"{at} is a {container} restricted by {extra}"]
    return _narrowed(spec, schema.get(values, True), tuple(inner), f"{at}{step}")


def _kind(schema: dict[str, Any], keywords: set[str]) -> str:
    if "properties" in schema:
        return "a model with fixed fields"
    if kind := schema.get("type"):
        return {"array": "a list", "object": "a map"}.get(kind, kind)
    return f"restricted by {sorted(keywords)}"


def _resolve(spec: dict[str, Any], ref: str) -> Any:
    node = spec
    for part in ref.removeprefix("#/").split("/"):
        node = node[part]
    return node


async def _no_content() -> None:
    return None
