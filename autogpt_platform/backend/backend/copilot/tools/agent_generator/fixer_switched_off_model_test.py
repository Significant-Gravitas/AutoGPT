"""Otto's agent generator builds no new work on a model the catalog has
switched off, while an edit keeps the model a node was saved on. The LLM
blocks' schema still lists switched-off models so saved workflows keep
validating. The tests run the generator's fix and validate path on the real
block schemas."""

import copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.codex import CodeGenerationBlock
from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.blocks.llm import DEFAULT_LLM_MODEL, AITextGeneratorBlock
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.fix_agent import FixAgentGraphTool
from backend.copilot.tools.models import (
    AgentPreviewResponse,
    AgentSavedResponse,
    FixResultResponse,
)
from backend.data.llm_registry import llm_models
from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.llm_models import LLMModel

from .fixer import AgentFixer
from .pipeline import fix_validate_and_save

PIPELINE = "backend.copilot.tools.agent_generator.pipeline"

WORKFLOW_ID = "0f0f0f0f-0000-4000-8000-000000000000"
INPUT_NODE = "11111111-1111-4111-8111-111111111111"
AI_NODE = "22222222-2222-4222-8222-222222222222"
OUTPUT_NODE = "33333333-3333-4333-8333-333333333333"
NEW_AI_NODE = "44444444-4444-4444-8444-444444444444"

ENABLED = {model.slug for model in get_catalog().models if model.is_enabled}
SWITCHED_OFF = sorted(
    model.slug
    for model in get_catalog().models
    if not model.is_enabled and model.slug in {m.value for m in LLMModel}
)


def workflow(ai_block: type, model: str) -> dict:
    """Input -> AI block on *model* -> output, as Otto writes one."""
    return {
        "id": WORKFLOW_ID,
        "name": "Haiku Maker",
        "description": "Writes a haiku about a topic.",
        "version": 1,
        "is_active": True,
        "nodes": [
            node(INPUT_NODE, AgentInputBlock().id, {"name": "topic"}),
            node(AI_NODE, ai_block().id, {"model": model}),
            node(OUTPUT_NODE, AgentOutputBlock().id, {"name": "haiku"}),
        ],
        "links": [
            link(INPUT_NODE, "result", AI_NODE, "prompt"),
            link(AI_NODE, "response", OUTPUT_NODE, "value"),
        ],
    }


def with_new_ai_node(agent_json: dict, model: str) -> dict:
    """*agent_json* plus an AI Text Generator node an edit adds on *model*."""
    added = node(
        NEW_AI_NODE,
        AITextGeneratorBlock().id,
        {"model": model, "prompt": "Write a haiku about rain."},
    )
    return {**agent_json, "nodes": [*agent_json["nodes"], added]}


def node(node_id: str, block_id: str, input_default: dict) -> dict:
    return {
        "id": node_id,
        "block_id": block_id,
        "input_default": input_default,
        "metadata": {"position": {"x": 0, "y": 0}},
    }


def link(source_id: str, source_name: str, sink_id: str, sink_name: str) -> dict:
    return {
        "source_id": source_id,
        "source_name": source_name,
        "sink_id": sink_id,
        "sink_name": sink_name,
    }


async def preview(agent_json: dict, **kwargs) -> AgentPreviewResponse:
    """The generator's real fix -> validate path, without saving."""
    result = await fix_validate_and_save(
        agent_json, user_id=None, session_id=None, save=False, **kwargs
    )
    assert isinstance(result, AgentPreviewResponse), result
    return result


def model_on(result: AgentPreviewResponse, node_id: str) -> str:
    return model_on_json(result.agent_json, node_id)


def model_on_json(agent_json: dict, node_id: str) -> str:
    return next(
        n["input_default"]["model"] for n in agent_json["nodes"] if n["id"] == node_id
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "switched_off, replacement",
    [
        ("gpt-5-2025-08-07", "gpt-6.1-sol"),  # OpenAI serves it until 11 Dec
        ("o3-2025-04-16", "gpt-6.1-sol"),
        ("claude-sonnet-4-5-20250929", "claude-sonnet-5-5"),
        ("mistralai/mistral-medium-3.1", "mistralai/mistral-medium-3-5"),
        ("o3-mini", "gpt-6.1-terra"),  # already shut down
    ],
)
async def test_a_new_workflow_on_a_switched_off_model_gets_its_replacement(
    switched_off: str, replacement: str
):
    """The replacement is the model's catalog fallback, which the boot pass
    and the retire CLI move stored nodes to and blocks run in its place."""
    result = await preview(workflow(AITextGeneratorBlock, switched_off))

    assert model_on(result, AI_NODE) == replacement


@pytest.mark.asyncio
@pytest.mark.parametrize("switched_off", SWITCHED_OFF)
async def test_no_new_workflow_is_left_on_a_switched_off_model(switched_off: str):
    result = await preview(workflow(AITextGeneratorBlock, switched_off))

    assert model_on(result, AI_NODE) in ENABLED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pinned", ["gpt-4o", "gpt-5.2-2025-12-11", "claude-opus-4-6", "gpt-6.1-sol"]
)
async def test_a_pinned_model_the_catalog_still_enables_is_kept(pinned: str):
    assert pinned in ENABLED
    result = await preview(workflow(AITextGeneratorBlock, pinned))

    assert model_on(result, AI_NODE) == pinned
    assert "retired model" not in result.message


@pytest.mark.asyncio
async def test_the_preview_tells_otto_which_model_replaced_the_one_it_wrote():
    """Otto wrote the retired model, so it would tell the user that one."""
    result = await preview(workflow(AITextGeneratorBlock, "gpt-5-2025-08-07"))

    assert "named a retired model" in result.message
    assert "uses 'gpt-6.1-sol' instead of 'gpt-5-2025-08-07'" in result.message


@pytest.mark.asyncio
async def test_an_edit_keeps_a_saved_switched_off_model_and_moves_new_work_off_it():
    """A node saved on a switched-off model keeps it, as the builder does, and
    runs on its fallback from the model's shutdown date. A node the edit adds
    on it gets the replacement."""
    saved = workflow(AITextGeneratorBlock, "gpt-5-2025-08-07")
    edited = with_new_ai_node(copy.deepcopy(saved), "gpt-5-2025-08-07")

    result = await preview(edited, saved_agent=saved)

    assert model_on(result, AI_NODE) == "gpt-5-2025-08-07"
    assert model_on(result, NEW_AI_NODE) == "gpt-6.1-sol"
    # The note for Otto names the node that moved, not the one that kept its
    # model.
    assert f"node '{NEW_AI_NODE}'" in result.message
    assert f"node '{AI_NODE}'" not in result.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "saved_model, written, kept",
    [
        ("gpt-5-2025-08-07", "openai/gpt-5-2025-08-07", "gpt-5-2025-08-07"),
        ("openai/gpt-5-2025-08-07", "gpt-5-2025-08-07", "gpt-5-2025-08-07"),
        ("gpt-6.1-sol", "openai/gpt-6.1-sol", "gpt-6.1-sol"),
    ],
    ids=[
        "switched-off-edit-adds-the-prefix",
        "switched-off-saved-with-the-prefix",
        "enabled-edit-adds-the-prefix",
    ],
)
async def test_an_edit_keeps_the_saved_model_in_any_spelling(
    saved_model: str, written: str, kept: str
):
    """The enum takes a provider-prefixed spelling too. An edit that names
    the model its node was saved on keeps that model, switched off or not,
    written as the block's schema lists it."""
    saved = workflow(AITextGeneratorBlock, saved_model)
    edited = workflow(AITextGeneratorBlock, written)

    result = await preview(edited, saved_agent=saved)

    assert model_on(result, AI_NODE) == kept


@pytest.mark.asyncio
async def test_an_edit_still_replaces_a_value_that_names_no_model():
    saved = workflow(AITextGeneratorBlock, "gpt-6.1-sol")
    edited = workflow(AITextGeneratorBlock, "gpt-6.1-sol-preview")

    result = await preview(edited, saved_agent=saved)

    assert model_on(result, AI_NODE) == DEFAULT_LLM_MODEL.value


@pytest.mark.asyncio
async def test_an_edit_keeps_no_saved_model_the_block_doesnt_offer():
    """The Codex block's own list doesn't offer gpt-6.1-sol, so a Codex node
    saved on it gets the block's default, in either spelling."""
    saved = workflow(CodeGenerationBlock, "gpt-6.1-sol")
    edited = workflow(CodeGenerationBlock, "openai/gpt-6.1-sol")

    result = await preview(edited, saved_agent=saved)

    assert model_on(result, AI_NODE) == "gpt-5.3-codex"


@pytest.mark.asyncio
async def test_an_edit_that_moves_a_node_onto_a_switched_off_model_gets_the_replacement():
    saved = workflow(AITextGeneratorBlock, "gpt-5.6-sol")
    edited = workflow(AITextGeneratorBlock, "o3-2025-04-16")

    result = await preview(edited, saved_agent=saved)

    assert model_on(result, AI_NODE) == "gpt-6.1-sol"


@pytest.mark.asyncio
async def test_fix_agent_graph_leaves_a_switched_off_model_to_the_save():
    """fix_agent_graph can't tell a new node from a saved one, so it keeps
    the model; create_agent or edit_agent decides when the workflow is saved."""
    result = await FixAgentGraphTool()._execute(
        user_id="test-user-switched-off-model",
        session=make_session("test-user-switched-off-model"),
        agent_json=workflow(AITextGeneratorBlock, "gpt-5-2025-08-07"),
    )

    assert isinstance(result, FixResultResponse)
    assert result.fixed_agent_json is not None
    nodes = {n["id"]: n for n in result.fixed_agent_json["nodes"]}
    assert nodes[AI_NODE]["input_default"]["model"] == "gpt-5-2025-08-07"


@pytest.mark.asyncio
async def test_a_switched_off_model_without_a_replacement_the_block_offers_gets_its_default():
    """The catalog sends gpt-5.1-codex to gpt-6.1-sol, which the Codex block's
    own list doesn't offer, so the node gets the block's default."""
    result = await preview(workflow(CodeGenerationBlock, "gpt-5.1-codex"))

    assert model_on(result, AI_NODE) == "gpt-5.3-codex"


@pytest.mark.asyncio
async def test_the_save_tells_otto_which_model_replaced_the_one_it_wrote():
    graph = MagicMock(id="graph-1", version=1)
    graph.name = "Haiku Maker"
    library_agent = MagicMock(id="library-1")
    save = AsyncMock(return_value=(graph, library_agent))

    with patch(f"{PIPELINE}.save_agent_to_library", save):
        result = await fix_validate_and_save(
            workflow(AITextGeneratorBlock, "gpt-5-2025-08-07"),
            user_id="test-user-switched-off-model",
            session_id=None,
        )

    assert isinstance(result, AgentSavedResponse)
    assert "uses 'gpt-6.1-sol' instead of 'gpt-5-2025-08-07'" in result.message
    saved_json = save.await_args.args[0]
    assert model_on_json(saved_json, AI_NODE) == "gpt-6.1-sol"


@pytest.mark.asyncio
async def test_a_fallback_the_catalog_switched_off_too_is_not_written(monkeypatch):
    """A chained retirement: the replacement was switched off later as well.
    The catalog's fallback map leaves such an entry out today; the fixer
    doesn't depend on that, so the node gets the block's default."""
    monkeypatch.setitem(
        llm_models.RETIRED_MODEL_FALLBACKS,
        LLMModel("o3-2025-04-16"),
        (LLMModel("gpt-5-2025-08-07"), None),
    )

    result = await preview(workflow(AITextGeneratorBlock, "o3-2025-04-16"))

    assert model_on(result, AI_NODE) == DEFAULT_LLM_MODEL.value


@pytest.mark.parametrize(
    "enum, written",
    [
        (None, "gpt-6.1-terra"),
        (["o3-mini", "gpt-5-2025-08-07", "gpt-6.1-sol"], "gpt-6.1-sol"),
    ],
    ids=["its-fallback", "the-first-model-still-on"],
)
def test_a_switched_off_block_default_is_not_written(
    enum: list[str] | None, written: str
):
    """A block whose default is switched off: a node with no model gets the
    default's fallback when the block offers it, else the first model the
    block lists that the catalog still has switched on."""
    block = copy.deepcopy(AITextGeneratorBlock().get_info().model_dump())
    model_schema = block["inputSchema"]["properties"]["model"]
    model_schema["default"] = "o3-mini"
    if enum is not None:
        model_schema["enum"] = enum
    agent = {"nodes": [node(AI_NODE, block["id"], {})], "links": []}

    fixed = AgentFixer().fix_ai_model_parameter(agent, [block])

    assert model_on_json(fixed, AI_NODE) == written
