"""Pins for the two promises of the second pass that no test of the head states:
"other inputs" means every input but the model, and what the inputs can't
place goes in the edit's order."""

import copy
import uuid

import pytest

from backend.blocks.llm import AITextGeneratorBlock

from .fixer_switched_off_model_test import (
    AI_NODE,
    GPT5,
    GPT5_FALLBACK,
    NEW_AI_NODE,
    model_on,
    preview,
    with_new_ai_node,
    workflow,
)
from .fixer_switched_off_saved_ids_test import (
    stored_by_the_library,
    with_new_ai_node_first,
)


@pytest.mark.asyncio
async def test_a_respelled_model_does_not_make_a_node_differ_from_its_saved_self():
    """The inputs a stored node is matched on exclude the model, so an edit that
    writes both nodes with a provider prefix still tells the saved node (same
    prompt) from the copy listed before it (another prompt)."""
    saved = workflow(AITextGeneratorBlock, GPT5)
    saved["nodes"][1]["input_default"]["prompt"] = "Write a haiku about snow."
    edited = with_new_ai_node_first(copy.deepcopy(saved), GPT5)
    for each in edited["nodes"]:
        if each["id"] in (AI_NODE, NEW_AI_NODE):
            each["input_default"]["model"] = f"openai/{GPT5}"

    result = await preview(edited, saved_agent=stored_by_the_library(saved))

    assert model_on(result, AI_NODE) == GPT5
    assert model_on(result, NEW_AI_NODE) == GPT5_FALLBACK


@pytest.mark.asyncio
async def test_stored_nodes_the_inputs_cant_place_go_to_the_edits_nodes_in_its_order():
    """Two saved nodes and three edited ones, none with inputs equal to a stored
    node's: the first two in the edit keep the model and the last is new work."""
    saved = with_new_ai_node(workflow(AITextGeneratorBlock, GPT5), GPT5)
    third = str(uuid.uuid4())
    edited = with_new_ai_node(copy.deepcopy(saved), GPT5, node_id=third)
    prompts = {AI_NODE: "first", NEW_AI_NODE: "second", third: "third"}
    for each in edited["nodes"]:
        if each["id"] in prompts:
            each["input_default"]["prompt"] = prompts[each["id"]]

    result = await preview(edited, saved_agent=stored_by_the_library(saved))

    assert model_on(result, AI_NODE) == GPT5
    assert model_on(result, NEW_AI_NODE) == GPT5
    assert model_on(result, third) == GPT5_FALLBACK
