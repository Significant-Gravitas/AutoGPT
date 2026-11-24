"""Matching a node to its saved self in the edit loop the library really runs.

``fixer_switched_off_saved_ids_test`` builds the stored version with
``stored_by_the_library(raw file)``. The library stores what the *fixer* wrote,
under new ids, and the fixer rewrites some inputs (single braces in a prompt
become double). The file Otto keeps editing still has the raw inputs, so the
"other inputs equal its own" preference has to compare like with like.

The second half pins the other side of the same rule: an edit made from the
current ids is told from Otto's own file under stale ids by carrying an id the
stored version has, so a node it replaces is new work."""

import copy

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
    with_node_id,
)


def saved_with_a_prompt_the_fixer_rewrites() -> dict:
    saved = workflow(AITextGeneratorBlock, GPT5)
    saved["nodes"][1]["input_default"]["prompt"] = "Write a haiku about {topic}"
    return saved


async def stored_after_the_first_edit(saved: dict) -> dict:
    """What the library stores once Otto's first ``edit_agent`` call (by the ids
    it read) went through the real pipeline: the fixer's output, new ids."""
    first = await preview(copy.deepcopy(saved), saved_agent=saved)
    stored = stored_by_the_library(first.agent_json)
    assert (
        stored["nodes"][1]["input_default"]["prompt"] == "Write a haiku about {{topic}}"
    )
    return stored


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "add", [with_new_ai_node, with_new_ai_node_first], ids=["copy-last", "copy-first"]
)
async def test_a_copy_is_new_work_when_the_fixer_rewrote_the_stored_inputs(add):
    """Otto's file still has ``{topic}``; the stored node has ``{{topic}}``. The
    copy has other inputs again, so neither equals the stored node and the edit's
    order decides: with the copy listed first it took the model, and the saved
    node was moved instead."""
    saved = saved_with_a_prompt_the_fixer_rewrites()
    stored = await stored_after_the_first_edit(saved)

    result = await preview(add(copy.deepcopy(saved), GPT5), saved_agent=stored)

    assert model_on(result, AI_NODE) == GPT5
    assert model_on(result, NEW_AI_NODE) == GPT5_FALLBACK


@pytest.mark.asyncio
async def test_a_node_replaced_in_an_edit_from_the_current_ids_is_new_work():
    """The edit carries the input and output nodes' ids, so it was made from the
    stored version's ids and its new node isn't the one it deleted."""
    saved = workflow(AITextGeneratorBlock, GPT5)

    result = await preview(
        with_node_id(saved, AI_NODE, NEW_AI_NODE), saved_agent=copy.deepcopy(saved)
    )

    assert model_on(result, NEW_AI_NODE) == GPT5_FALLBACK


@pytest.mark.asyncio
async def test_a_node_replaced_in_an_edit_under_stale_ids_keeps_the_model():
    """Under stale ids no node of the edit is known, which is also what Otto's own
    file looks like: nothing tells a replaced node from a kept one, and keeping
    the saved model is the safe side."""
    saved = workflow(AITextGeneratorBlock, GPT5)

    result = await preview(
        with_node_id(saved, AI_NODE, NEW_AI_NODE),
        saved_agent=stored_by_the_library(saved),
    )

    assert model_on(result, NEW_AI_NODE) == GPT5
