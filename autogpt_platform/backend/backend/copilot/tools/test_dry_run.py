"""Tests for dry-run execution mode."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.helpers import execute_block
from backend.copilot.tools.models import BlockOutputResponse, ErrorResponse
from backend.copilot.tools.run_block import RunBlockTool
from backend.executor.simulator import (
    build_simulation_prompt,
    prepare_dry_run,
    simulate_block,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_block(
    name: str = "TestBlock",
    description: str = "A test block",
    input_props: dict | None = None,
    output_props: dict | None = None,
):
    """Create a minimal mock block with jsonschema() methods."""
    block = MagicMock()
    block.name = name
    block.description = description

    in_props = input_props or {"query": {"type": "string"}}
    out_props = output_props or {
        "result": {"type": "string"},
        "error": {"type": "string"},
    }

    block.input_schema = MagicMock()
    block.input_schema.jsonschema.return_value = {
        "type": "object",
        "properties": in_props,
        "required": list(in_props.keys()),
    }
    block.input_schema.get_credentials_fields.return_value = {}
    block.input_schema.get_credentials_fields_info.return_value = {}

    block.output_schema = MagicMock()
    block.output_schema.jsonschema.return_value = {
        "type": "object",
        "properties": out_props,
        "required": ["result"],
    }

    return block


def make_openai_response(
    content: str, prompt_tokens: int = 100, completion_tokens: int = 50
):
    """Build a mock OpenAI chat completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


# ---------------------------------------------------------------------------
# simulate_block tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simulate_block_basic():
    """simulate_block returns correct (output_name, output_data) tuples.

    Empty error pins should be omitted (not yielded) — only pins with
    meaningful values are forwarded.
    """
    mock_block = make_mock_block()
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=make_openai_response('{"result": "simulated output", "error": ""}')
    )

    with patch(
        "backend.executor.simulator.get_openai_client", return_value=mock_client
    ) as mock_get_client:
        outputs = []
        async for name, data in simulate_block(mock_block, {"query": "test"}):
            outputs.append((name, data))

    mock_get_client.assert_called_once_with(prefer_openrouter=True)
    assert ("result", "simulated output") in outputs
    # Empty error pin should NOT be yielded — the simulator omits empty values
    assert ("error", "") not in outputs


@pytest.mark.asyncio
async def test_simulate_block_json_retry():
    """LLM returns invalid JSON twice then valid; verifies 3 total calls."""
    mock_block = make_mock_block()
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            make_openai_response("not json at all"),
            make_openai_response("still not json"),
            make_openai_response('{"result": "ok", "error": ""}'),
        ]
    )

    with patch(
        "backend.executor.simulator.get_openai_client", return_value=mock_client
    ):
        outputs = []
        async for name, data in simulate_block(mock_block, {"query": "test"}):
            outputs.append((name, data))

    assert mock_client.chat.completions.create.call_count == 3
    assert ("result", "ok") in outputs
    # Empty error pin is dropped
    assert ("error", "") not in outputs


@pytest.mark.asyncio
async def test_simulate_block_all_retries_exhausted():
    """LLM always returns invalid JSON; verify error tuple is yielded."""
    mock_block = make_mock_block()
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=make_openai_response("bad json !!!")
    )

    with patch(
        "backend.executor.simulator.get_openai_client", return_value=mock_client
    ):
        outputs = []
        async for name, data in simulate_block(mock_block, {"query": "test"}):
            outputs.append((name, data))

    # All retry attempts should have been consumed
    assert mock_client.chat.completions.create.call_count == 5  # _MAX_JSON_RETRIES
    assert len(outputs) == 1
    name, data = outputs[0]
    assert name == "error"
    assert "[SIMULATOR ERROR" in data


@pytest.mark.asyncio
async def test_simulate_block_missing_output_pins():
    """LLM response missing some output pins; they are omitted (not yielded)."""
    mock_block = make_mock_block(
        output_props={
            "result": {"type": "string"},
            "count": {"type": "integer"},
            "error": {"type": "string"},
        }
    )
    mock_client = AsyncMock()
    # Only returns "result", missing "count" and "error"
    mock_client.chat.completions.create = AsyncMock(
        return_value=make_openai_response('{"result": "hello"}')
    )

    with patch(
        "backend.executor.simulator.get_openai_client", return_value=mock_client
    ):
        outputs = {}
        async for name, data in simulate_block(mock_block, {"query": "hi"}):
            outputs[name] = data

    assert outputs["result"] == "hello"
    # Missing pins are omitted — only pins with meaningful values are yielded
    assert "count" not in outputs
    assert "error" not in outputs


@pytest.mark.asyncio
async def test_simulate_block_no_client():
    """When no OpenAI client is available, yields SIMULATOR ERROR."""
    mock_block = make_mock_block()

    with patch("backend.executor.simulator.get_openai_client", return_value=None):
        outputs = []
        async for name, data in simulate_block(mock_block, {}):
            outputs.append((name, data))

    assert len(outputs) == 1
    name, data = outputs[0]
    assert name == "error"
    assert "[SIMULATOR ERROR" in data


@pytest.mark.asyncio
async def test_simulate_block_truncates_long_inputs():
    """Inputs with very long strings should be truncated in the prompt."""
    mock_block = make_mock_block(input_props={"text": {"type": "string"}})
    long_text = "x" * 30000  # 30k chars, above the 20k threshold

    system_prompt, user_prompt = build_simulation_prompt(
        mock_block, {"text": long_text}
    )

    # The user prompt should contain TRUNCATED marker
    assert "[TRUNCATED]" in user_prompt
    # And the total length of the value in the prompt should be well under 30k chars
    parsed = json.loads(user_prompt.split("## Current Inputs\n", 1)[1])
    assert len(parsed["text"]) < 25000


def test_build_simulation_prompt_lists_available_output_pins():
    """The prompt should list available output pins (excluding error) so the LLM
    knows which keys it MUST include.  Error is excluded because the prompt
    tells the LLM to omit it unless simulating a logical failure."""
    block = make_mock_block()  # default output_props has "result" and "error"
    system_prompt, _ = build_simulation_prompt(block, {"query": "test"})
    available_line = [
        line for line in system_prompt.splitlines() if "Available output pins" in line
    ][0]
    assert '"result"' in available_line
    # "error" is intentionally excluded from the required output pins list
    # since the prompt instructs the LLM to omit it unless simulating errors
    assert '"error"' not in available_line


# ---------------------------------------------------------------------------
# execute_block dry-run tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_block_dry_run_skips_real_execution():
    """execute_block(dry_run=True) calls simulate_block, NOT block.execute."""
    mock_block = make_mock_block()
    mock_block.execute = AsyncMock()  # should NOT be called

    async def fake_simulate(block, input_data):
        yield "result", "simulated"

    # Patching at helpers.simulate_block works because helpers.py imports
    # simulate_block at the top of the module. If the import were lazy
    # (inside the function), we'd need to patch the source module instead.
    with patch(
        "backend.copilot.tools.helpers.simulate_block", side_effect=fake_simulate
    ):
        response = await execute_block(
            block=mock_block,
            block_id="test-block-id",
            input_data={"query": "hello"},
            user_id="user-1",
            session_id="session-1",
            node_exec_id="node-exec-1",
            matched_credentials={},
            dry_run=True,
        )

    mock_block.execute.assert_not_called()
    assert isinstance(response, BlockOutputResponse)
    assert response.success is True


@pytest.mark.asyncio
async def test_execute_block_dry_run_response_format():
    """Dry-run response should look like a normal success (no dry-run signal to LLM)."""
    mock_block = make_mock_block()

    async def fake_simulate(block, input_data):
        yield "result", "simulated"

    with patch(
        "backend.copilot.tools.helpers.simulate_block", side_effect=fake_simulate
    ):
        response = await execute_block(
            block=mock_block,
            block_id="test-block-id",
            input_data={"query": "hello"},
            user_id="user-1",
            session_id="session-1",
            node_exec_id="node-exec-1",
            matched_credentials={},
            dry_run=True,
        )

    assert isinstance(response, BlockOutputResponse)
    assert "[DRY RUN]" not in response.message
    assert "executed successfully" in response.message
    assert response.success is True
    assert response.outputs == {"result": ["simulated"]}
    # is_dry_run is present in model_dump (used by frontend SSE via StreamToolOutputAvailable).
    # tool_adapter._truncating strips it from the LLM-facing result AFTER stashing,
    # so the frontend receives it but the LLM does not.
    assert response.is_dry_run is True
    assert "is_dry_run" in response.model_dump()
    # model_dump_json excludes None fields — is_dry_run=True must still appear.
    assert '"is_dry_run"' in response.model_dump_json(exclude_none=True)


@pytest.mark.asyncio
async def test_execute_block_normal_run_omits_is_dry_run():
    """Normal (non-dry-run) BlockOutputResponse must NOT carry is_dry_run in JSON.

    The field must be absent so the frontend and LLM don't see spurious
    'is_dry_run: false' noise on every tool call.
    """
    from backend.copilot.tools.models import BlockOutputResponse

    response = BlockOutputResponse(
        message="Block 'X' executed successfully",
        block_id="b1",
        block_name="X",
        outputs={"result": ["real"]},
        success=True,
        session_id="s1",
        # is_dry_run intentionally NOT set → stays None
    )

    assert response.is_dry_run is None
    serialized = response.model_dump_json(exclude_none=True)
    assert '"is_dry_run"' not in serialized


@pytest.mark.asyncio
async def test_execute_block_real_execution_unchanged():
    """dry_run=False should still go through the real execution path."""
    mock_block = make_mock_block()

    # We expect it to hit the real path, which will fail on workspace_db() call.
    # Just verify simulate_block is NOT called.
    simulate_called = False

    async def fake_simulate(block, input_data):
        nonlocal simulate_called
        simulate_called = True
        yield "result", "should not happen"

    with patch(
        "backend.copilot.tools.helpers.simulate_block", side_effect=fake_simulate
    ):
        with patch(
            "backend.copilot.tools.helpers.workspace_db",
            side_effect=Exception("db not available"),
        ):
            response = await execute_block(
                block=mock_block,
                block_id="test-block-id",
                input_data={"query": "hello"},
                user_id="user-1",
                session_id="session-1",
                node_exec_id="node-exec-1",
                matched_credentials={},
                dry_run=False,
            )

    assert simulate_called is False
    # The real path raised an exception, so we get an ErrorResponse (which has .error attr)
    assert hasattr(response, "error")


# ---------------------------------------------------------------------------
# RunBlockTool parameter tests
# ---------------------------------------------------------------------------


def test_run_block_tool_no_dry_run_param():
    """RunBlockTool parameters must NOT expose 'dry_run' — it's a session-level flag."""
    tool = RunBlockTool()
    params = tool.parameters
    assert "dry_run" not in params["properties"]


@pytest.mark.asyncio
async def test_run_block_tool_uses_session_dry_run():
    """RunBlockTool._execute derives dry_run from session.dry_run, not from kwargs.

    Behavioral test: intercepts prepare_block_for_execution and captures the
    dry_run argument actually passed, asserting it matches session.dry_run
    regardless of what a hypothetical kwarg would say.
    """
    from backend.copilot.tools.run_block import RunBlockTool

    tool = RunBlockTool()
    session = MagicMock()
    session.dry_run = True
    session.session_id = "test-session-id"

    captured = {}

    async def capture_prep(**kwargs):
        captured["dry_run"] = kwargs.get("dry_run")
        # Return an error response to short-circuit execution
        from backend.copilot.tools.models import ErrorResponse

        return ErrorResponse(message="stub", session_id="test-session-id")

    with patch(
        "backend.copilot.tools.run_block.prepare_block_for_execution",
        side_effect=capture_prep,
    ):
        await tool._execute(
            user_id="user-1",
            session=session,
            block_id="block-id-123",
            input_data={},
        )

    # dry_run must come from session, not from a default or kwarg
    assert captured["dry_run"] is True


@pytest.mark.asyncio
async def test_run_block_tool_uses_session_dry_run_false():
    """RunBlockTool._execute passes dry_run=False when session.dry_run is False.

    Symmetric counterpart to test_run_block_tool_uses_session_dry_run (True case).
    """
    from backend.copilot.tools.run_block import RunBlockTool

    tool = RunBlockTool()
    session = MagicMock()
    session.dry_run = False
    session.session_id = "test-session-id"

    captured = {}

    async def capture_prep(**kwargs):
        captured["dry_run"] = kwargs.get("dry_run")
        from backend.copilot.tools.models import ErrorResponse

        return ErrorResponse(message="stub", session_id="test-session-id")

    with patch(
        "backend.copilot.tools.run_block.prepare_block_for_execution",
        side_effect=capture_prep,
    ):
        await tool._execute(
            user_id="user-1",
            session=session,
            block_id="block-id-123",
            input_data={},
        )

    assert captured["dry_run"] is False


@pytest.mark.asyncio
async def test_execute_block_dry_run_no_empty_error_from_simulator():
    """The simulator no longer yields empty error pins, so execute_block
    simply passes through whatever the simulator produces.

    Since the fix is at the simulator level, even if a simulator somehow
    yields only non-error outputs, they pass through unchanged.
    """
    mock_block = make_mock_block()

    async def fake_simulate(block, input_data):
        # Simulator now omits empty error pins at source
        yield "result", "simulated output"

    with patch(
        "backend.copilot.tools.helpers.simulate_block", side_effect=fake_simulate
    ):
        response = await execute_block(
            block=mock_block,
            block_id="test-block-id",
            input_data={"query": "hello"},
            user_id="user-1",
            session_id="session-1",
            node_exec_id="node-exec-1",
            matched_credentials={},
            dry_run=True,
        )

    assert isinstance(response, BlockOutputResponse)
    assert response.success is True
    assert response.is_dry_run is True
    assert "error" not in response.outputs
    assert response.outputs == {"result": ["simulated output"]}


@pytest.mark.asyncio
async def test_execute_block_dry_run_keeps_nonempty_error_pin():
    """Dry-run should keep the 'error' pin when it contains a real error message."""
    mock_block = make_mock_block()

    async def fake_simulate(block, input_data):
        yield "result", ""
        yield "error", "API rate limit exceeded"

    with patch(
        "backend.copilot.tools.helpers.simulate_block", side_effect=fake_simulate
    ):
        response = await execute_block(
            block=mock_block,
            block_id="test-block-id",
            input_data={"query": "hello"},
            user_id="user-1",
            session_id="session-1",
            node_exec_id="node-exec-1",
            matched_credentials={},
            dry_run=True,
        )

    assert isinstance(response, BlockOutputResponse)
    assert response.success is True
    # Non-empty error should be preserved
    assert "error" in response.outputs
    assert response.outputs["error"] == ["API rate limit exceeded"]


@pytest.mark.asyncio
async def test_execute_block_dry_run_message_includes_completed_status():
    """Dry-run message should clearly indicate COMPLETED status."""
    mock_block = make_mock_block()

    async def fake_simulate(block, input_data):
        yield "result", "simulated"

    with patch(
        "backend.copilot.tools.helpers.simulate_block", side_effect=fake_simulate
    ):
        response = await execute_block(
            block=mock_block,
            block_id="test-block-id",
            input_data={"query": "hello"},
            user_id="user-1",
            session_id="session-1",
            node_exec_id="node-exec-1",
            matched_credentials={},
            dry_run=True,
        )

    assert isinstance(response, BlockOutputResponse)
    assert "executed successfully" in response.message


@pytest.mark.asyncio
async def test_execute_block_dry_run_simulator_error_returns_error_response():
    """When simulate_block yields a SIMULATOR ERROR tuple, execute_block returns ErrorResponse."""
    mock_block = make_mock_block()

    async def fake_simulate_error(block, input_data):
        yield (
            "error",
            "[SIMULATOR ERROR — NOT A BLOCK FAILURE] No LLM client available (missing OpenAI/OpenRouter API key).",
        )

    with patch(
        "backend.copilot.tools.helpers.simulate_block", side_effect=fake_simulate_error
    ):
        response = await execute_block(
            block=mock_block,
            block_id="test-block-id",
            input_data={"query": "hello"},
            user_id="user-1",
            session_id="session-1",
            node_exec_id="node-exec-1",
            matched_credentials={},
            dry_run=True,
        )

    assert isinstance(response, ErrorResponse)
    assert "[SIMULATOR ERROR" in response.message


# ---------------------------------------------------------------------------
# prepare_dry_run tests
# ---------------------------------------------------------------------------


def test_prepare_dry_run_orchestrator_block():
    """prepare_dry_run caps iterations and overrides model to simulation model."""
    from backend.blocks.orchestrator import OrchestratorBlock

    block = OrchestratorBlock()
    input_data = {"prompt": "hello", "model": "gpt-4o", "agent_mode_max_iterations": 10}
    with patch(
        "backend.executor.simulator._get_platform_openrouter_key",
        return_value="sk-or-test-key",
    ):
        result = prepare_dry_run(block, input_data)

    assert result is not None
    # Model is overridden to the simulation model (not the user's model).
    assert result["model"] != "gpt-4o"
    assert result["agent_mode_max_iterations"] == 1
    assert result["_dry_run_api_key"] == "sk-or-test-key"
    # Original input_data should not be mutated.
    assert input_data["model"] == "gpt-4o"


def test_prepare_dry_run_agent_executor_block():
    """prepare_dry_run returns a copy of input_data for AgentExecutorBlock.

    AgentExecutorBlock must execute for real during dry-run so it can spawn
    a child graph execution (whose blocks are then simulated).  Its Output
    schema has no properties, so LLM simulation would yield zero outputs.
    """
    from backend.blocks.agent import AgentExecutorBlock

    block = AgentExecutorBlock()
    input_data = {
        "user_id": "u1",
        "graph_id": "g1",
        "graph_version": 1,
        "inputs": {"text": "hello"},
        "input_schema": {},
        "output_schema": {},
    }
    result = prepare_dry_run(block, input_data)

    assert result is not None
    # Input data is returned as-is (no model swap needed).
    assert result["user_id"] == "u1"
    assert result["graph_id"] == "g1"
    # Original input_data should not be mutated.
    assert result is not input_data


def test_prepare_dry_run_regular_block_returns_none():
    """prepare_dry_run returns None for a regular block (use simulator)."""
    mock_block = make_mock_block()
    assert prepare_dry_run(mock_block, {"query": "test"}) is None


# ---------------------------------------------------------------------------
# Input/output block passthrough tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simulate_agent_input_block_passthrough():
    """AgentInputBlock should pass through the value directly, no LLM call."""
    from backend.blocks.io import AgentInputBlock

    block = AgentInputBlock()
    outputs = []
    async for name, data in simulate_block(
        block, {"value": "hello world", "name": "q"}
    ):
        outputs.append((name, data))

    assert outputs == [("result", "hello world")]


@pytest.mark.asyncio
async def test_simulate_agent_dropdown_input_block_passthrough():
    """AgentDropdownInputBlock (subclass of AgentInputBlock) should pass through."""
    from backend.blocks.io import AgentDropdownInputBlock

    block = AgentDropdownInputBlock()
    outputs = []
    async for name, data in simulate_block(
        block,
        {
            "value": "Option B",
            "name": "sev",
            "options": ["Option A", "Option B"],
        },
    ):
        outputs.append((name, data))

    assert outputs == [("result", "Option B")]


@pytest.mark.asyncio
async def test_simulate_agent_input_block_none_value_falls_back_to_name():
    """AgentInputBlock with value=None falls back to the input name."""
    from backend.blocks.io import AgentInputBlock

    block = AgentInputBlock()
    outputs = []
    async for name, data in simulate_block(block, {"value": None, "name": "q"}):
        outputs.append((name, data))

    # When value is None, the simulator falls back to the "name" field
    assert outputs == [("result", "q")]


@pytest.mark.asyncio
async def test_simulate_agent_output_block_passthrough():
    """AgentOutputBlock should pass through value as output."""
    from backend.blocks.io import AgentOutputBlock

    block = AgentOutputBlock()
    outputs = []
    async for name, data in simulate_block(
        block, {"value": "result text", "name": "out1"}
    ):
        outputs.append((name, data))

    assert ("output", "result text") in outputs
    assert ("name", "out1") in outputs


@pytest.mark.asyncio
async def test_simulate_agent_output_block_no_name():
    """AgentOutputBlock without name in input should still yield output."""
    from backend.blocks.io import AgentOutputBlock

    block = AgentOutputBlock()
    outputs = []
    async for name, data in simulate_block(block, {"value": 42}):
        outputs.append((name, data))

    assert outputs == [("output", 42)]


# ---------------------------------------------------------------------------
# RunAgentTool session-level dry_run override tests
# ---------------------------------------------------------------------------


def _make_dry_run_session(dry_run: bool = True) -> MagicMock:
    """Return a minimal ChatSession mock with dry_run set."""
    session = MagicMock()
    session.dry_run = dry_run
    session.session_id = "test-session-id"
    session.successful_agent_runs = {}
    return session


def _make_graph_mock(graph_id: str = "g1") -> MagicMock:
    """Return a minimal GraphModel mock."""
    graph = MagicMock()
    graph.id = graph_id
    graph.name = "Test Agent"
    graph.version = 1
    graph.description = "A test agent"
    graph.input_schema = {"type": "object", "properties": {}, "required": []}
    graph.credentials_input_schema = {"type": "object", "properties": {}}
    graph.trigger_setup_info = None
    return graph


@pytest.mark.asyncio
async def test_run_agent_session_dry_run_overrides_kwargs():
    """session.dry_run=True must override any dry_run=False from LLM kwargs.

    The LLM can pass dry_run=False, but when the session is a dry-run session,
    the session-level flag wins and forces dry_run=True for all runs.
    """
    from backend.copilot.tools.run_agent import RunAgentTool

    tool = RunAgentTool()
    session = _make_dry_run_session(dry_run=True)
    graph = _make_graph_mock()

    captured_params = {}

    async def capture_prerequisites(graph, user_id, params, session_id):
        captured_params["dry_run"] = params.dry_run
        return {}, None

    with (
        patch(
            "backend.copilot.tools.run_agent.fetch_graph_from_store_slug",
            new_callable=AsyncMock,
            return_value=(graph, None),
        ),
        patch.object(tool, "_check_prerequisites", side_effect=capture_prerequisites),
        patch.object(tool, "_run_agent", new_callable=AsyncMock) as mock_run_agent,
    ):
        mock_run_agent.return_value = MagicMock()

        # Pass dry_run=False in kwargs — session.dry_run=True should win.
        await tool._execute(
            user_id="user-1",
            session=session,
            username_agent_slug="user/agent",
            dry_run=False,  # LLM would pass this; session should override it
        )

    # Session-level flag must have overridden the False kwarg
    assert captured_params["dry_run"] is True


@pytest.mark.asyncio
async def test_run_agent_session_dry_run_false_allows_scheduling():
    """session.dry_run=False must pass dry_run=False through to _check_prerequisites.

    Verifies that when the session is not a dry run, params.dry_run=False
    is what reaches the prerequisite check — not some stale True value.
    """
    from backend.copilot.tools.run_agent import RunAgentTool

    tool = RunAgentTool()
    session = _make_dry_run_session(dry_run=False)
    graph = _make_graph_mock()

    captured_params = {}

    async def capture_prerequisites(graph, user_id, params, session_id):
        captured_params["dry_run"] = params.dry_run
        return {}, None

    with (
        patch(
            "backend.copilot.tools.run_agent.fetch_graph_from_store_slug",
            new_callable=AsyncMock,
            return_value=(graph, None),
        ),
        patch.object(tool, "_check_prerequisites", side_effect=capture_prerequisites),
        patch.object(tool, "_schedule_agent", new_callable=AsyncMock) as mock_schedule,
    ):
        mock_schedule.return_value = MagicMock()

        await tool._execute(
            user_id="user-1",
            session=session,
            username_agent_slug="user/agent",
            schedule_name="daily",
            cron="0 9 * * *",
        )

    # Non-dry-run session must propagate dry_run=False
    assert captured_params["dry_run"] is False


@pytest.mark.asyncio
async def test_run_agent_session_dry_run_false_allows_llm_dry_run_true():
    """session.dry_run=False must NOT override an explicit dry_run=True kwarg.

    In a normal session the LLM can still request a dry run on individual
    run_agent calls (e.g. "test this agent without executing it").
    Only session.dry_run=True forces dry_run — the False value does not force
    the opposite direction.
    """
    from backend.copilot.tools.run_agent import RunAgentTool

    tool = RunAgentTool()
    session = _make_dry_run_session(dry_run=False)
    graph = _make_graph_mock()

    captured_params = {}

    async def capture_prerequisites(graph, user_id, params, session_id):
        captured_params["dry_run"] = params.dry_run
        return {}, None

    with (
        patch(
            "backend.copilot.tools.run_agent.fetch_graph_from_store_slug",
            new_callable=AsyncMock,
            return_value=(graph, None),
        ),
        patch.object(tool, "_check_prerequisites", side_effect=capture_prerequisites),
        patch.object(tool, "_run_agent", new_callable=AsyncMock) as mock_run_agent,
    ):
        mock_run_agent.return_value = MagicMock()

        # LLM passes dry_run=True; normal session must NOT override it to False
        await tool._execute(
            user_id="user-1",
            session=session,
            username_agent_slug="user/agent",
            dry_run=True,
        )

    # LLM-requested dry_run=True is preserved in a normal session
    assert captured_params["dry_run"] is True
