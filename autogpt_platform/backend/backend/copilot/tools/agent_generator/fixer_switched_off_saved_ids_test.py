"""An edit keeps a node's switched-off model, though the library stores every
saved version under new node ids.

``update_graph_in_library`` runs ``reassign_ids`` on each version it stores, so
the file Otto keeps editing carries the ids of the version it read, and the
stored one carries new ones. Matching saved nodes by id alone makes the second
``edit_agent`` call of a save, dry-run, fix loop treat every node as new."""

import copy
import uuid

import pytest

from backend.blocks.llm import AIStructuredResponseGeneratorBlock, AITextGeneratorBlock
from backend.copilot.tools.agent_generator.core import graph_to_json, json_to_graph
from backend.data.graph import make_graph_model

from . import switched_off_models
from .fixer_switched_off_model_test import (
    AI_NODE,
    GPT5,
    GPT5_FALLBACK,
    NEW_AI_NODE,
    O3,
    fallback_of,
    model_on,
    node,
    preview,
    with_new_ai_node,
    workflow,
)

COPY_NODE = "55555555-5555-4555-8555-555555555555"


def stored_by_the_library(agent_json: dict) -> dict:
    """The graph ``update_graph_in_library`` persists for *agent_json*: the same
    nodes under new node ids. ``graph_to_json`` drops the links, so this is
    for the stored side only."""
    graph = make_graph_model(json_to_graph(copy.deepcopy(agent_json)), "user")
    graph.reassign_ids(user_id="user", reassign_graph_id=False)
    return graph_to_json(graph)


def with_fresh_ids(agent_json: dict) -> dict:
    """*agent_json* with every node under a new id, the links following."""
    rewritten = copy.deepcopy(agent_json)
    new_ids = {n["id"]: str(uuid.uuid4()) for n in rewritten["nodes"]}
    for n in rewritten["nodes"]:
        n["id"] = new_ids[n["id"]]
    for link in rewritten["links"]:
        link["source_id"] = new_ids[link["source_id"]]
        link["sink_id"] = new_ids[link["sink_id"]]
    return rewritten


def models(result) -> list[str]:
    return sorted(
        n["input_default"]["model"]
        for n in result.agent_json["nodes"]
        if n["block_id"] == AITextGeneratorBlock().id
    )


@pytest.mark.asyncio
async def test_a_second_edit_keeps_a_saved_switched_off_model():
    """Otto's file still has the ids of the version it read, and the version its
    first edit stored has new ones."""
    first_edit = workflow(AITextGeneratorBlock, GPT5)
    stored = stored_by_the_library(first_edit)
    assert not {n["id"] for n in stored["nodes"]} & {
        n["id"] for n in first_edit["nodes"]
    }

    result = await preview(copy.deepcopy(first_edit), saved_agent=stored)

    assert model_on(result, AI_NODE) == GPT5
    assert "switched-off model" not in result.message


@pytest.mark.asyncio
async def test_an_edit_that_gives_every_node_a_fresh_id_keeps_the_saved_model():
    saved = workflow(AITextGeneratorBlock, GPT5)

    result = await preview(with_fresh_ids(saved), saved_agent=saved)

    assert models(result) == [GPT5]


def with_new_ai_node_first(agent_json: dict, model: str) -> dict:
    """*agent_json* plus an AI Text Generator node an edit adds on *model*,
    listed before every other node."""
    *nodes, added = with_new_ai_node(agent_json, model)["nodes"]
    return {**agent_json, "nodes": [added, *nodes]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored", [copy.deepcopy, stored_by_the_library], ids=["ids-current", "ids-stale"]
)
@pytest.mark.parametrize(
    "add", [with_new_ai_node, with_new_ai_node_first], ids=["copy-last", "copy-first"]
)
async def test_a_copy_of_a_saved_node_is_new_work_whichever_ids_the_edit_carries(
    stored, add
):
    """Under stale ids the copy and the node it copies compete for one stored
    node; the copy's new inputs tell them apart, wherever it sits."""
    saved = workflow(AITextGeneratorBlock, GPT5)

    result = await preview(add(copy.deepcopy(saved), GPT5), saved_agent=stored(saved))

    assert model_on(result, AI_NODE) == GPT5
    assert model_on(result, NEW_AI_NODE) == GPT5_FALLBACK


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored", [copy.deepcopy, stored_by_the_library], ids=["ids-current", "ids-stale"]
)
async def test_of_two_identical_nodes_one_keeps_each_stored_model(stored):
    """Nothing tells an exact copy from the node it copies: the first in the
    edit keeps the model, and the copy is new work."""
    saved = with_new_ai_node(workflow(AITextGeneratorBlock, GPT5), GPT5)
    edited = with_new_ai_node(copy.deepcopy(saved), GPT5, node_id=COPY_NODE)

    result = await preview(edited, saved_agent=stored(saved))

    assert model_on(result, AI_NODE) == GPT5
    assert model_on(result, NEW_AI_NODE) == GPT5
    assert model_on(result, COPY_NODE) == GPT5_FALLBACK


@pytest.mark.asyncio
async def test_two_identical_saved_nodes_both_keep_their_model_under_stale_ids():
    """Each stored node pairs with its own edited node, so the second of two
    equal nodes isn't left over and moved."""
    saved = with_new_ai_node(
        with_new_ai_node(workflow(AITextGeneratorBlock, GPT5), GPT5),
        GPT5,
        node_id=COPY_NODE,
    )

    result = await preview(
        copy.deepcopy(saved), saved_agent=stored_by_the_library(saved)
    )

    assert model_on(result, AI_NODE) == GPT5
    assert model_on(result, NEW_AI_NODE) == GPT5
    assert model_on(result, COPY_NODE) == GPT5


@pytest.mark.asyncio
async def test_a_saved_node_an_edit_gave_new_inputs_keeps_its_model_under_stale_ids():
    """Other inputs only break ties: a prompt an edit changes doesn't make
    the node new work, beside a saved node on the same model it left as is."""
    saved = with_new_ai_node(workflow(AITextGeneratorBlock, GPT5), GPT5)
    edited = copy.deepcopy(saved)
    edited["nodes"][-1]["input_default"]["prompt"] = "Write a haiku about snow."

    result = await preview(edited, saved_agent=stored_by_the_library(saved))

    assert model_on(result, AI_NODE) == GPT5
    assert model_on(result, NEW_AI_NODE) == GPT5
    assert "switched-off model" not in result.message


@pytest.mark.asyncio
async def test_a_node_on_another_block_or_model_is_not_the_saved_one():
    saved = stored_by_the_library(workflow(AITextGeneratorBlock, GPT5))
    other_model = workflow(AITextGeneratorBlock, O3)

    result = await preview(other_model, saved_agent=saved)

    assert model_on(result, AI_NODE) == fallback_of(O3)


def moved_to_the_structured_response_block(agent_json: dict) -> dict:
    swapped = copy.deepcopy(agent_json)
    ai = swapped["nodes"][1]
    ai["block_id"] = AIStructuredResponseGeneratorBlock().id
    ai["input_default"]["expected_format"] = {"haiku": "the haiku"}
    return swapped


@pytest.mark.asyncio
async def test_a_node_moved_to_another_block_under_its_id_is_new_work():
    saved = workflow(AITextGeneratorBlock, GPT5)

    result = await preview(
        moved_to_the_structured_response_block(saved), saved_agent=saved
    )

    assert model_on(result, AI_NODE) == GPT5_FALLBACK


@pytest.mark.asyncio
async def test_a_node_moved_onto_a_deleted_node_s_block_does_not_take_its_model():
    """The edit moves the text node onto the structured block under its own
    id and deletes the structured node: a known id on another block is
    changed work, not the deleted node."""
    saved = workflow(AITextGeneratorBlock, GPT5)
    structured = node(
        NEW_AI_NODE,
        AIStructuredResponseGeneratorBlock().id,
        {
            "model": GPT5,
            "prompt": "Write a haiku about rain.",
            "expected_format": {"haiku": "the haiku"},
        },
    )

    result = await preview(
        moved_to_the_structured_response_block(saved),
        saved_agent={**saved, "nodes": [*saved["nodes"], structured]},
    )

    assert model_on(result, AI_NODE) == GPT5_FALLBACK


@pytest.mark.asyncio
async def test_a_stored_node_on_another_block_does_not_stand_in_for_a_new_one():
    saved = workflow(AITextGeneratorBlock, GPT5)

    result = await preview(
        moved_to_the_structured_response_block(saved),
        saved_agent=stored_by_the_library(saved),
    )

    assert model_on(result, AI_NODE) == GPT5_FALLBACK


def with_node_id(agent_json: dict, old_id: str, new_id: str) -> dict:
    """*agent_json* with node *old_id* deleted and an identical one added under
    *new_id*, the links following."""
    rewritten = copy.deepcopy(agent_json)
    for n in rewritten["nodes"]:
        if n["id"] == old_id:
            n["id"] = new_id
    for link in rewritten["links"]:
        for end in ("source_id", "sink_id"):
            if link[end] == old_id:
                link[end] = new_id
    return rewritten


@pytest.mark.asyncio
async def test_a_changed_node_does_not_take_the_stored_model_of_the_one_that_kept_it():
    """Both nodes carry stale ids: the first was moved to another model, the
    second still has the saved one."""
    saved = stored_by_the_library(workflow(AITextGeneratorBlock, GPT5))
    edited = with_new_ai_node(workflow(AITextGeneratorBlock, O3), GPT5)

    result = await preview(edited, saved_agent=saved)

    assert model_on(result, AI_NODE) == fallback_of(O3)
    assert model_on(result, NEW_AI_NODE) == GPT5


def test_a_large_pool_pairs_in_one_pass(mocker):
    """Pairing reads each node's inputs once, so an edit of thousands of nodes
    on one block and model under stale ids doesn't stall the event loop."""
    count = 2000
    block_id = AITextGeneratorBlock().id

    def nodes(prefix: str, prompts) -> dict:
        return {
            "nodes": [
                {
                    "id": f"{prefix}{i}",
                    "block_id": block_id,
                    "input_default": {"model": GPT5, "prompt": prompt},
                }
                for i, prompt in enumerate(prompts)
            ]
        }

    stored = nodes("stored-", [f"prompt {i}" for i in range(count)])
    edited = nodes("edited-", [f"prompt {i}" for i in reversed(range(count))])
    read_inputs = mocker.spy(switched_off_models, "_inputs")

    matched = switched_off_models.models_by_node(edited, stored)

    assert matched == {f"edited-{i}": GPT5 for i in range(count)}
    assert read_inputs.call_count == 2 * count
