"""Fixtures for the pai engine's unit tests.

These tests drive the real agent loop against Pydantic AI's ``FunctionModel``
and mock every I/O boundary (Redis, DB, the tool registry's execution), so
they need no running services: the root conftest's session-wide test server
is replaced by a no-op here.
"""

from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from openai.types.chat import ChatCompletionToolParam
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatSession

from .events import PaiEventMapper
from .model import PaiRoute
from .runner import RunInputs
from .state import PaiTurnState
from .toolset import RegistryToolset

StreamFn = Callable[[list[ModelMessage], AgentInfo], AsyncIterator[Any]]


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    """Override the root conftest's DB-backed cleanup: nothing here needs it."""
    yield


def make_session() -> ChatSession:
    return ChatSession.new("user-1", dry_run=False, session_id="sess-1")


def echo_tool(name: str = "web_fetch") -> ChatCompletionToolParam:
    return ChatCompletionToolParam(
        type="function",
        function={
            "name": name,
            "description": f"{name} for tests",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    )


def make_inputs(
    stream_fn: StreamFn,
    state: PaiTurnState,
    *,
    user_prompt: Any = "hello",
    history: list[ModelMessage] | None = None,
    deferred_results: Any = None,
    tools: list[ChatCompletionToolParam] | None = None,
    max_rounds: int = 10,
) -> RunInputs:
    agent: Agent[None, str | DeferredToolRequests] = Agent(
        FunctionModel(stream_function=stream_fn),
        instructions=["STATIC", lambda: "<turn_context>dyn</turn_context>"],
        output_type=[str, DeferredToolRequests],
    )
    return RunInputs(
        agent=agent,
        user_prompt=user_prompt,
        history=history or [],
        deferred_results=deferred_results,
        model_settings={},
        toolset=RegistryToolset(
            tools if tools is not None else [echo_tool()],
            sink=state,
            user_id="user-1",
            disabled_groups=[],
            disabled_tools=frozenset(),
        ),
        mapper=PaiEventMapper(state.emit, state.session_messages),
        route=PaiRoute(
            model="anthropic/claude-test", source="env", provider="openrouter"
        ),
        config=ChatConfig(),
        session_id="sess-1",
        turn_start=0,
        max_rounds=max_rounds,
    )


@pytest.fixture
def state() -> PaiTurnState:
    return PaiTurnState(
        make_session(), model="anthropic/claude-test", routing_source="env"
    )
