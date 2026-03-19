"""Tests for dry-run execution mode."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.run_block import RunBlockTool


# Lazy imports used inside tests to avoid Pyright "unresolved import" on the
# simulator module, which lives in this worktree only.
def _get_simulate_block():
    from backend.executor.simulator import simulate_block  # noqa: PLC0415

    return simulate_block


def _get_build_simulation_prompt():
    from backend.executor.simulator import build_simulation_prompt  # noqa: PLC0415

    return build_simulation_prompt


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
    """simulate_block returns correct (output_name, output_data) tuples."""
    mock_block = make_mock_block()
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=make_openai_response('{"result": "simulated output", "error": ""}')
    )

    simulate_block = _get_simulate_block()
    with patch(
        "backend.executor.simulator.get_openai_client", return_value=mock_client
    ):
        outputs = []
        async for name, data in simulate_block(mock_block, {"query": "test"}):
            outputs.append((name, data))

    assert ("result", "simulated output") in outputs
    assert ("error", "") in outputs


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

    simulate_block = _get_simulate_block()
    with patch(
        "backend.executor.simulator.get_openai_client", return_value=mock_client
    ):
        outputs = []
        async for name, data in simulate_block(mock_block, {"query": "test"}):
            outputs.append((name, data))

    assert mock_client.chat.completions.create.call_count == 3
    assert ("result", "ok") in outputs


@pytest.mark.asyncio
async def test_simulate_block_all_retries_exhausted():
    """LLM always returns invalid JSON; verify error tuple is yielded."""
    mock_block = make_mock_block()
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=make_openai_response("bad json !!!")
    )

    simulate_block = _get_simulate_block()
    with patch(
        "backend.executor.simulator.get_openai_client", return_value=mock_client
    ):
        outputs = []
        async for name, data in simulate_block(mock_block, {"query": "test"}):
            outputs.append((name, data))

    assert len(outputs) == 1
    name, data = outputs[0]
    assert name == "error"
    assert "[SIMULATOR ERROR" in data


@pytest.mark.asyncio
async def test_simulate_block_missing_output_pins():
    """LLM response missing some output pins; verify they're filled with None."""
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

    simulate_block = _get_simulate_block()
    with patch(
        "backend.executor.simulator.get_openai_client", return_value=mock_client
    ):
        outputs = dict()
        async for name, data in simulate_block(mock_block, {"query": "hi"}):
            outputs[name] = data

    assert outputs["result"] == "hello"
    assert outputs["count"] is None  # missing pin filled with None
    assert outputs["error"] == ""  # "error" pin filled with ""


@pytest.mark.asyncio
async def test_simulate_block_no_client():
    """When no OpenAI client is available, yields SIMULATOR ERROR."""
    mock_block = make_mock_block()

    simulate_block = _get_simulate_block()
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

    build_simulation_prompt = _get_build_simulation_prompt()
    system_prompt, user_prompt = build_simulation_prompt(
        mock_block, {"text": long_text}
    )

    # The user prompt should contain TRUNCATED marker
    assert "[TRUNCATED]" in user_prompt
    # And the total length of the value in the prompt should be well under 30k chars
    parsed = json.loads(user_prompt.split("## Current Inputs\n", 1)[1])
    assert len(parsed["text"]) < 25000


# ---------------------------------------------------------------------------
# execute_block dry-run tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_block_dry_run_skips_real_execution():
    """execute_block(dry_run=True) calls simulate_block, NOT block.execute."""
    from backend.copilot.tools.helpers import execute_block

    mock_block = make_mock_block()
    mock_block.execute = AsyncMock()  # should NOT be called

    async def fake_simulate(block, input_data):
        yield "result", "simulated"

    with patch("backend.executor.simulator.simulate_block", side_effect=fake_simulate):
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
    from backend.copilot.tools.models import BlockOutputResponse

    assert isinstance(response, BlockOutputResponse)
    assert response.success is True


@pytest.mark.asyncio
async def test_execute_block_dry_run_response_format():
    """Dry-run response should contain [DRY RUN] in message and success=True."""
    from backend.copilot.tools.helpers import execute_block

    mock_block = make_mock_block()

    async def fake_simulate(block, input_data):
        yield "result", "simulated"

    with patch("backend.executor.simulator.simulate_block", side_effect=fake_simulate):
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

    from backend.copilot.tools.models import BlockOutputResponse

    assert isinstance(response, BlockOutputResponse)
    assert "[DRY RUN]" in response.message
    assert response.success is True
    assert response.outputs == {"result": ["simulated"]}


@pytest.mark.asyncio
async def test_execute_block_real_execution_unchanged():
    """dry_run=False should still go through the real execution path."""
    from backend.copilot.tools.helpers import execute_block

    mock_block = make_mock_block()

    # We expect it to hit the real path, which will fail on workspace_db() call.
    # Just verify simulate_block is NOT called.
    simulate_called = False

    async def fake_simulate(block, input_data):
        nonlocal simulate_called
        simulate_called = True
        yield "result", "should not happen"

    with patch("backend.executor.simulator.simulate_block", side_effect=fake_simulate):
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
    """RunBlockTool parameters should include 'dry_run'."""
    tool = RunBlockTool()
    params = tool.parameters
    assert "dry_run" in params["properties"]
    assert params["properties"]["dry_run"]["type"] == "boolean"


def test_run_block_tool_dry_run_calls_execute():
    """RunBlockTool._execute extracts dry_run from kwargs correctly.

    We verify the extraction logic directly by inspecting the source, then confirm
    the kwarg is forwarded in the execute_block call site.
    """
    import inspect

    import backend.copilot.tools.run_block as run_block_module

    source = inspect.getsource(run_block_module.RunBlockTool._execute)
    # Verify dry_run is extracted from kwargs
    assert "dry_run" in source
    assert 'kwargs.get("dry_run"' in source or 'kwargs.get("dry_run"' in source

    source_execute = inspect.getsource(run_block_module)
    # Verify dry_run is passed through to execute_block call
    assert "dry_run=dry_run" in source_execute
