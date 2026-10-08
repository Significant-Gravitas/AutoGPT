"""The registry adapter exposes the baseline's tool surface, schemas unchanged."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from backend.copilot.baseline.service import _filter_tools_by_permissions
from backend.copilot.capabilities.sources import EAGER_CORE
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.tools import TOOL_REGISTRY, get_available_tools

from .conftest import make_session
from .state import PaiTurnState
from .toolset import RegistryToolset, held_review_id, tool_definitions

SAMPLE = ["find_capability", "run_capability", "bash_exec", "web_fetch", "TodoWrite"]


def _toolset(tools, state: PaiTurnState) -> RegistryToolset:
    return RegistryToolset(
        tools,
        sink=state,
        user_id="user-1",
        disabled_groups=[],
        disabled_tools=frozenset(),
    )


def test_definitions_carry_each_schema_unchanged():
    tools = [TOOL_REGISTRY[name].as_openai_tool() for name in SAMPLE]
    definitions = {d.name: d for d in tool_definitions(tools)}
    for name in SAMPLE:
        tool = TOOL_REGISTRY[name]
        assert definitions[name].parameters_json_schema == tool.parameters
        assert definitions[name].description == tool.description


def test_exposure_is_the_baselines_eager_core():
    names = {d.name for d in tool_definitions(get_available_tools())}
    assert names <= EAGER_CORE
    assert "run_capability" in names
    # Deferred registry tools are reached through run_capability only.
    assert "create_agent" not in names


def test_permissions_filter_applies_before_exposure():
    tools = _filter_tools_by_permissions(
        get_available_tools(),
        CopilotPermissions(tools=["web_fetch"], tools_exclude=False),
    )
    assert [d.name for d in tool_definitions(tools)] == ["web_fetch"]


@pytest.mark.asyncio
async def test_model_receives_the_same_schemas():
    state = PaiTurnState(make_session(), model="m", routing_source="env")
    tools = [TOOL_REGISTRY[name].as_openai_tool() for name in SAMPLE]
    received: dict[str, dict] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.update({t.name: t.parameters_json_schema for t in info.function_tools})
        return ModelResponse(parts=[TextPart(content="ok")])

    agent = Agent(FunctionModel(respond), toolsets=[_toolset(tools, state)])
    await agent.run("hi")
    for name in SAMPLE:
        assert received[name] == TOOL_REGISTRY[name].parameters


@pytest.mark.asyncio
async def test_calls_run_through_execute_tool_with_the_turn_gates():
    state = PaiTurnState(make_session(), model="m", routing_source="env")
    result = StreamToolOutputAvailable(
        toolCallId="c1", toolName="web_fetch", output="x"
    )
    execute = AsyncMock(return_value=result)
    toolset = RegistryToolset(
        [TOOL_REGISTRY["web_fetch"].as_openai_tool()],
        sink=state,
        user_id="user-1",
        disabled_groups=["graphiti"],
        disabled_tools=frozenset({"hire_expert"}),
    )

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("web_fetch", {"url": "https://x"}, tool_call_id="c1")
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    with patch("backend.copilot.pai.toolset.execute_tool", execute):
        await Agent(FunctionModel(respond), toolsets=[toolset]).run("go")

    kwargs = execute.await_args.kwargs if execute.await_args else {}
    assert kwargs["tool_name"] == "web_fetch"
    assert kwargs["parameters"] == {"url": "https://x"}
    assert kwargs["tool_call_id"] == "c1"
    assert kwargs["disabled_groups"] == ["graphiti"]
    assert kwargs["disabled_tools"] == frozenset({"hire_expert"})
    assert state.emitted == [result]
    assert state.tool_persistence.results["c1"].content == "x"


def test_only_a_refusal_with_a_review_is_a_held_call():
    held = StreamToolOutputAvailable(
        toolCallId="c",
        toolName="t",
        success=False,
        output='{"type": "approval_required", "message": "m", "tool_name": "t", '
        '"reason": "r", "review_id": "rev-1"}',
    )
    final = held.model_copy(
        update={
            "output": '{"type": "approval_required", "message": "m", '
            '"tool_name": "t", "reason": "declined", "review_id": null}'
        }
    )
    assert held_review_id(held) == "rev-1"
    assert held_review_id(final) is None
    ok = held.model_copy(update={"success": True})
    assert held_review_id(ok) is None
