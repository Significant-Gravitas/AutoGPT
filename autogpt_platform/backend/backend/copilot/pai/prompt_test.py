"""The behaviour contract and where it sits in the pai instructions."""

import re

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from backend.copilot.prompting import NO_REPLY, approval_mode_supplement
from backend.copilot.tools import TOOL_REGISTRY

from .prompt import (
    TURN_CONTEXT_TAG,
    PromptInputs,
    behaviour_contract,
    build_static_instructions,
    build_turn_context,
)

RULE_LABELS = (
    "Reply first",
    "Status beats",
    "Default to action",
    "Initiative, once",
    "Recurring means routine",
    "Background work",
    "Tone",
    "Memory",
)
WORD_BUDGET = 900
BASE = "BASE SYSTEM PROMPT SENTINEL"


def _inputs(**overrides) -> PromptInputs:
    values = dict(
        base_system_prompt=BASE,
        graphiti_enabled=True,
        experts_enabled=True,
        expert_id=None,
        source_platform=None,
        autopilot_mode="auto",
        builder_session_suffix="\n<builder_session>graph</builder_session>",
        expert_session_suffix="\n<expert_identity>expert</expert_identity>",
    )
    values.update(overrides)
    return PromptInputs(**values)


def test_contract_is_non_empty_and_static():
    first = behaviour_contract()
    assert first.strip()
    assert all(behaviour_contract() == first for _ in range(3))


def test_contract_fits_the_word_budget():
    assert len(behaviour_contract().split()) <= WORD_BUDGET


@pytest.mark.parametrize("label", RULE_LABELS)
def test_contract_has_every_rule_label(label: str):
    assert f"**{label}.**" in behaviour_contract()


def test_contract_has_no_template_slots_or_per_user_values():
    text = behaviour_contract()
    assert "{" not in text and "}" not in text
    assert not re.search(r"<[^>\n]+>", text), "no tags or <placeholder>s"
    for token in ("user_id", "session_id", "expert_id", "$", "%s", "TODO"):
        assert token not in text


def test_contract_names_only_real_tools():
    named = set(re.findall(r"`(?:tool:)?([A-Za-z_]+)`", behaviour_contract()))
    named.discard(NO_REPLY)
    assert named, "the contract should point at concrete tools"
    assert named <= set(TOOL_REGISTRY), named - set(TOOL_REGISTRY)


def test_contract_silence_token_matches_the_chat_platform_rule():
    assert f"`{NO_REPLY}`" in behaviour_contract()


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"autopilot_mode": None, "graphiti_enabled": False},
        {"experts_enabled": False, "source_platform": "discord"},
        {"expert_id": "expert-1", "autopilot_mode": "ask_first"},
    ],
)
def test_contract_closes_the_static_instructions(overrides: dict):
    inputs = _inputs(**overrides)
    static = build_static_instructions(inputs)
    contract = behaviour_contract()

    assert static.startswith(BASE)
    assert static.endswith(contract)
    assert static.count(contract) == 1
    contract_at = static.index(contract)
    for earlier in (
        approval_mode_supplement(inputs.autopilot_mode),
        inputs.builder_session_suffix,
        inputs.expert_session_suffix,
    ):
        if earlier:
            assert static.index(earlier) < contract_at
    assert f"<{TURN_CONTEXT_TAG}>\n" not in static


async def test_model_sees_base_then_contract_then_turn_context():
    static = build_static_instructions(_inputs())
    turn_context = build_turn_context(["<budget_status>low</budget_status>"])
    seen: list[str] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        seen.append(request.instructions or "")
        return ModelResponse(parts=[TextPart("ok")])

    def dynamic() -> str:
        return turn_context

    agent = Agent(FunctionModel(model_fn), instructions=[static, dynamic])
    await agent.run("hello")

    sent = seen[0]
    base_at = sent.index(BASE)
    contract_at = sent.index(behaviour_contract())
    turn_at = sent.index(turn_context)
    assert base_at < contract_at < turn_at
