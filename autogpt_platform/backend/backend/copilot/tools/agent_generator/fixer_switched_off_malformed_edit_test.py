"""A node whose ``input_default`` is missing or isn't an object doesn't stop the
fixer on an edit under stale ids. The second pass reads the other inputs of every
node of a block, input and output nodes included, and the validator is the one to
report a malformed node (``fixer_switched_off_pins_test`` covers
``fix_agent_graph``, where every id is known and the second pass never runs)."""

import copy

import pytest

from backend.blocks.llm import AITextGeneratorBlock

from .blocks import get_blocks_as_dicts
from .fixer import AgentFixer
from .fixer_switched_off_model_test import (
    AI_NODE,
    GPT5,
    INPUT_NODE,
    model_on_json,
    workflow,
)
from .fixer_switched_off_saved_ids_test import stored_by_the_library

MISSING = object()


@pytest.mark.parametrize(
    "input_default",
    [MISSING, "not an object", ["a", "list"]],
    ids=["missing", "text", "list"],
)
def test_a_stale_edit_with_a_node_that_has_no_usable_input_default_is_still_fixed(
    input_default,
):
    saved = workflow(AITextGeneratorBlock, GPT5)
    edited = copy.deepcopy(saved)
    node = next(n for n in edited["nodes"] if n["id"] == INPUT_NODE)
    if input_default is MISSING:
        del node["input_default"]
    else:
        node["input_default"] = input_default

    fixed = AgentFixer().apply_all_fixes(
        edited, get_blocks_as_dicts(), saved_agent=stored_by_the_library(saved)
    )

    assert model_on_json(fixed, AI_NODE) == GPT5
