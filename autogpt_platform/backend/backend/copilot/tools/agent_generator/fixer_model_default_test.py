"""The building guide tells Otto to leave ``model`` unset unless the user names
one, which only works while the fixer fills in the defaults the guide promises."""

from pathlib import Path

from backend.blocks.llm import DEFAULT_LLM_MODEL, AITextGeneratorBlock
from backend.blocks.orchestrator import OrchestratorBlock

from .fixer import ORCHESTRATOR_DEFAULT_MODEL, AgentFixer
from .helpers import generate_uuid


def _fix(block_cls: type, input_default: dict) -> dict:
    block = block_cls().get_info().model_dump()
    agent = {
        "id": generate_uuid(),
        "nodes": [
            {
                "id": generate_uuid(),
                "block_id": block["id"],
                "input_default": input_default,
                "metadata": {"position": {"x": 0, "y": 0}},
            }
        ],
        "links": [],
    }
    return AgentFixer().apply_all_fixes(agent, [block])["nodes"][0]["input_default"]


def test_guide_tells_otto_to_leave_model_unset():
    guide = (Path(__file__).parents[2] / "sdk" / "agent_generation_guide.md").read_text(
        encoding="utf-8"
    )
    guide = " ".join(guide.split())
    assert "leave `model` out of `input_default` unless the user names" in guide
    assert "keep a node's existing `model` unless the user asks" in guide


def test_unset_model_gets_recommended_default():
    assert _fix(AITextGeneratorBlock, {})["model"] == DEFAULT_LLM_MODEL.value


def test_unset_model_on_orchestrator_gets_its_own_default():
    assert _fix(OrchestratorBlock, {})["model"] == ORCHESTRATOR_DEFAULT_MODEL


def test_model_the_user_named_is_kept():
    block = AITextGeneratorBlock().get_info().model_dump()
    named = next(
        model
        for model in block["inputSchema"]["properties"]["model"]["enum"]
        if model != DEFAULT_LLM_MODEL.value
    )
    assert _fix(AITextGeneratorBlock, {"model": named})["model"] == named
