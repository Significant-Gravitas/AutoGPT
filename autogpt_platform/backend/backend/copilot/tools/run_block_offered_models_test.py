"""The cut of an LLM model list to the models the block picker offers must reach
every place a schema can put one, and nowhere else."""

import copy
from typing import Any

import pytest

from backend.copilot.tools.run_block import _offered_models_only

OFFERED = "gpt-4o"
SWITCHED_OFF = "o3-mini"


def llm_model_list(**overrides: Any) -> dict[str, Any]:
    node: dict[str, Any] = {
        "type": "string",
        "enum": [OFFERED, SWITCHED_OFF],
        "llm_model": True,
        "llm_model_metadata": {OFFERED: {"title": "GPT-4o"}},
    }
    return {**node, **overrides}


def enums(node: Any) -> list[list[str]]:
    found: list[list[str]] = []
    if isinstance(node, dict):
        if isinstance(node.get("enum"), list):
            found.append(node["enum"])
        for value in node.values():
            found += enums(value)
    elif isinstance(node, list):
        for item in node:
            found += enums(item)
    return found


@pytest.mark.parametrize(
    "schema",
    [
        {"properties": {"model": llm_model_list()}},
        {"properties": {"model": {"anyOf": [llm_model_list(), {"type": "null"}]}}},
        {"properties": {"model": {"oneOf": [llm_model_list(), {"type": "integer"}]}}},
        {"properties": {"models": {"type": "array", "items": llm_model_list()}}},
        {
            "properties": {
                "cfg": {"type": "object", "properties": {"m": llm_model_list()}}
            }
        },
        {
            "$defs": {"LLMModel": llm_model_list()},
            "properties": {"model": {"$ref": "#/$defs/LLMModel"}},
        },
    ],
    ids=["plain", "anyOf", "oneOf", "array items", "nested object", "$defs"],
)
def test_the_cut_reaches_every_place_a_schema_puts_an_llm_model_list(schema):
    original = copy.deepcopy(schema)

    cut = _offered_models_only(schema)

    assert enums(cut) == [[OFFERED]]
    assert schema == original  # the builder's cached schema is never touched


@pytest.mark.parametrize(
    "model",
    [
        {"type": "string", "enum": [OFFERED, SWITCHED_OFF]},
        {"enum": [OFFERED, SWITCHED_OFF], "llm_model_metadata": {OFFERED: {}}},
        {"enum": [OFFERED, SWITCHED_OFF], "llm_model": True},
        {
            "enum": [OFFERED, SWITCHED_OFF],
            "llm_model": True,
            "llm_model_metadata": None,
        },
        {"llm_model": True, "llm_model_metadata": {OFFERED: {}}},
        {"llm_model": True, "llm_model_metadata": {OFFERED: {}}, "enum": None},
    ],
    ids=[
        "a block's own list",
        "metadata without the llm_model flag",
        "flag without metadata",
        "metadata that is not a dict",
        "no enum",
        "enum that is not a list",
    ],
)
def test_a_model_list_without_the_picker_metadata_is_left_whole(model):
    schema = {"properties": {"model": model}}

    assert _offered_models_only(schema) == schema
