"""Tests for dry-run execution mode."""

import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import backend.copilot.tools.run_block as run_block_module
from backend.copilot.tools.helpers import execute_block
from backend.copilot.tools.models import BlockOutputResponse, ErrorResponse
from backend.copilot.tools.run_block import RunBlockTool
from backend.executor.simulator import (
    _build_mcp_simulation_prompt,
    build_simulation_prompt,
    simulate_block,
    simulate_mcp_block,
)

# NOTE: simulate_block delegates to simulate_mcp_block internally for
# MCPToolBlock, but we keep the direct import for targeted tests.

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

    Empty "error" pins are dropped at source — only non-empty errors are yielded.
    """
    mock_block = make_mock_block()
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=make_openai_response('{"result": "simulated output", "error": ""}')
    )

    with patch(
        "backend.executor.simulator.get_openai_client", return_value=mock_client
    ):
        outputs = []
        async for name, data in simulate_block(mock_block, {"query": "test"}):
            outputs.append((name, data))

    assert ("result", "simulated output") in outputs
    # Empty error pin is dropped at the simulator level
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
    """LLM response missing some output pins; verify non-error pins filled with None."""
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
    assert outputs["count"] is None  # missing pin filled with None
    assert "error" not in outputs  # missing error pin is omitted entirely


@pytest.mark.asyncio
async def test_simulate_block_keeps_nonempty_error():
    """simulate_block keeps non-empty error pins (simulated logical errors)."""
    mock_block = make_mock_block()
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=make_openai_response(
            '{"result": "", "error": "API rate limit exceeded"}'
        )
    )

    with patch(
        "backend.executor.simulator.get_openai_client", return_value=mock_client
    ):
        outputs = []
        async for name, data in simulate_block(mock_block, {"query": "test"}):
            outputs.append((name, data))

    assert ("result", "") in outputs
    assert ("error", "API rate limit exceeded") in outputs


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


def test_build_simulation_prompt_excludes_error_from_must_include():
    """The 'MUST include' prompt line should NOT list 'error' — the prompt
    already instructs the LLM to OMIT error unless simulating a logical error.
    Including it in 'MUST include' would be contradictory."""
    block = make_mock_block()  # default output_props has "result" and "error"
    system_prompt, _ = build_simulation_prompt(block, {"query": "test"})
    must_include_line = [
        line for line in system_prompt.splitlines() if "MUST include" in line
    ][0]
    assert '"result"' in must_include_line
    assert '"error"' not in must_include_line


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
    """Dry-run response should match real execution message format and have success=True."""
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
    assert "[DRY RUN]" not in response.message  # must not leak to LLM context
    assert response.success is True
    assert response.outputs == {"result": ["simulated"]}


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


def test_run_block_tool_dry_run_param():
    """RunBlockTool parameters should include 'dry_run' as a required field."""
    tool = RunBlockTool()
    params = tool.parameters
    assert "dry_run" in params["properties"]
    assert params["properties"]["dry_run"]["type"] == "boolean"
    assert "dry_run" in params["required"]


def test_run_block_tool_dry_run_calls_execute():
    """RunBlockTool._execute accepts dry_run as a typed parameter.

    We verify the parameter exists in the signature and is forwarded to
    execute_block.
    """
    source = inspect.getsource(run_block_module.RunBlockTool._execute)
    # Verify dry_run is a typed parameter (not extracted from kwargs)
    assert "dry_run" in source
    assert "dry_run: bool" in source

    # Scope to _execute method source only — module-wide search is brittle
    # and can match unrelated text/comments.
    source_execute = inspect.getsource(run_block_module.RunBlockTool._execute)
    # Verify dry_run is passed through to execute_block call
    assert "dry_run=dry_run" in source_execute


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
# simulate_mcp_block tests
# ---------------------------------------------------------------------------


def test_build_mcp_simulation_prompt_contains_tool_info():
    """MCP simulation prompt should include tool name, schema, and arguments."""
    input_data = {
        "server_url": "https://mcp.example.com/mcp",
        "selected_tool": "get_weather",
        "tool_input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "tool_arguments": {"city": "London"},
    }

    system_prompt, user_prompt = _build_mcp_simulation_prompt(input_data)

    assert "get_weather" in system_prompt
    # Verify the full server URL is included (not just a substring)
    assert input_data["server_url"] in system_prompt
    assert '"city"' in system_prompt  # schema
    assert "London" in user_prompt  # arguments


def test_build_mcp_simulation_prompt_handles_empty_schema():
    """MCP prompt handles missing/empty tool_input_schema gracefully."""
    input_data = {
        "selected_tool": "my_tool",
        "tool_input_schema": {},
        "tool_arguments": {},
    }

    system_prompt, user_prompt = _build_mcp_simulation_prompt(input_data)

    assert "my_tool" in system_prompt
    assert "(none)" in system_prompt


def test_build_mcp_simulation_prompt_includes_description():
    """MCP prompt includes tool_description when present."""
    input_data = {
        "selected_tool": "search_tickets",
        "tool_description": "Search Linear tickets by query. Returns matching issues.",
        "tool_input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
        "tool_arguments": {"query": "login bug"},
    }
    system_prompt, _ = _build_mcp_simulation_prompt(input_data)
    assert "Search Linear tickets" in system_prompt
    assert "search_tickets" in system_prompt


@pytest.mark.asyncio
async def test_simulate_mcp_block_basic():
    """simulate_mcp_block returns result and error tuples."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=make_openai_response(
            '{"result": {"temperature": 22, "condition": "sunny"}, "error": ""}'
        )
    )

    input_data = {
        "server_url": "https://mcp.example.com/mcp",
        "selected_tool": "get_weather",
        "tool_input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "tool_arguments": {"city": "London"},
    }

    with patch(
        "backend.executor.simulator.get_openai_client", return_value=mock_client
    ):
        outputs = []
        async for name, data in simulate_mcp_block(None, input_data):
            outputs.append((name, data))

    assert len(outputs) == 2
    result_outputs = [d for n, d in outputs if n == "result"]
    assert result_outputs[0]["temperature"] == 22
    error_outputs = [d for n, d in outputs if n == "error"]
    assert error_outputs[0] == ""


@pytest.mark.asyncio
async def test_simulate_mcp_block_no_client():
    """When no OpenAI client is available, yields SIMULATOR ERROR."""
    with patch("backend.executor.simulator.get_openai_client", return_value=None):
        outputs = []
        async for name, data in simulate_mcp_block(None, {}):
            outputs.append((name, data))

    assert len(outputs) == 1
    assert outputs[0][0] == "error"
    assert "[SIMULATOR ERROR" in outputs[0][1]


@pytest.mark.asyncio
async def test_simulate_mcp_block_retries_on_bad_json():
    """simulate_mcp_block retries on invalid JSON, then succeeds."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            make_openai_response("not json"),
            make_openai_response('{"result": "ok", "error": ""}'),
        ]
    )

    with patch(
        "backend.executor.simulator.get_openai_client", return_value=mock_client
    ):
        outputs = []
        async for name, data in simulate_mcp_block(None, {"selected_tool": "test"}):
            outputs.append((name, data))

    assert mock_client.chat.completions.create.call_count == 2
    assert ("result", "ok") in outputs
