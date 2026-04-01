"""Tests for the LLM-powered block simulator (dry-run execution).

Covers:
  - Prompt building (credential stripping, realistic-output instructions)
  - Input/output block passthrough
  - prepare_dry_run routing
  - simulate_block output-pin filling
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.executor.simulator import (
    _truncate_input_values,
    _truncate_value,
    build_simulation_prompt,
    prepare_dry_run,
    simulate_block,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(
    *,
    name: str = "TestBlock",
    description: str = "A test block.",
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Any:
    """Create a minimal mock block for testing."""
    block = MagicMock()
    block.name = name
    block.description = description
    block.input_schema.jsonschema.return_value = input_schema or {
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    block.output_schema.jsonschema.return_value = output_schema or {
        "properties": {
            "result": {"type": "string"},
            "error": {"type": "string"},
        },
        "required": ["result"],
    }
    return block


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_short_string_unchanged(self) -> None:
        assert _truncate_value("hello") == "hello"

    def test_long_string_truncated(self) -> None:
        long_str = "x" * 30000
        result = _truncate_value(long_str)
        assert result.endswith("... [TRUNCATED]")
        assert len(result) < 25000

    def test_nested_dict_truncation(self) -> None:
        data = {"key": "y" * 30000}
        result = _truncate_input_values(data)
        assert result["key"].endswith("... [TRUNCATED]")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestBuildSimulationPrompt:
    def test_system_prompt_contains_block_name(self) -> None:
        block = _make_block(name="WebSearchBlock")
        system, _user = build_simulation_prompt(block, {"query": "test"})
        assert "WebSearchBlock" in system

    def test_system_prompt_contains_realistic_instruction(self) -> None:
        block = _make_block()
        system, _ = build_simulation_prompt(block, {})
        assert "REALISTIC" in system
        assert "Never return empty strings" in system

    def test_system_prompt_contains_no_auth_failure_instruction(self) -> None:
        block = _make_block()
        system, _ = build_simulation_prompt(block, {})
        assert "Do not simulate auth failures" in system

    def test_credentials_stripped_from_user_prompt(self) -> None:
        block = _make_block()
        _, user = build_simulation_prompt(
            block,
            {
                "query": "test",
                "credentials": {"api_key": "sk-secret"},
                "api_key": "sk-secret",
                "token": "tok-secret",
                "secret": "shh",
                "normal_field": "visible",
            },
        )
        assert "sk-secret" not in user
        assert "tok-secret" not in user
        assert "shh" not in user
        assert "visible" in user

    def test_error_pin_always_empty_instruction(self) -> None:
        block = _make_block()
        system, _ = build_simulation_prompt(block, {})
        assert "error" in system.lower()
        assert "empty string" in system.lower()

    def test_output_pin_names_in_prompt(self) -> None:
        block = _make_block(
            output_schema={
                "properties": {
                    "url": {"type": "string"},
                    "status_code": {"type": "integer"},
                },
            }
        )
        system, _ = build_simulation_prompt(block, {})
        assert "url" in system
        assert "status_code" in system


# ---------------------------------------------------------------------------
# prepare_dry_run routing
# ---------------------------------------------------------------------------


class TestPrepareDryRun:
    def test_orchestrator_block_caps_iterations(self) -> None:
        from backend.blocks.orchestrator import OrchestratorBlock

        block = OrchestratorBlock()
        result = prepare_dry_run(
            block, {"agent_mode_max_iterations": 10, "other": "val"}
        )
        assert result is not None
        assert result["agent_mode_max_iterations"] == 1
        assert result["other"] == "val"

    def test_orchestrator_block_zero_stays_zero(self) -> None:
        from backend.blocks.orchestrator import OrchestratorBlock

        block = OrchestratorBlock()
        result = prepare_dry_run(block, {"agent_mode_max_iterations": 0})
        assert result is not None
        assert result["agent_mode_max_iterations"] == 0

    def test_agent_executor_block_passthrough(self) -> None:
        from backend.blocks.agent import AgentExecutorBlock

        block = AgentExecutorBlock()
        result = prepare_dry_run(block, {"graph_id": "abc"})
        assert result is not None
        assert result["graph_id"] == "abc"

    def test_agent_executor_block_returns_identical_copy(self) -> None:
        """AgentExecutorBlock must execute for real during dry-run so it can
        spawn a child graph execution.  ``prepare_dry_run`` returns a shallow
        copy of input_data with no modifications -- every key/value must be
        identical, but the returned dict must be a *different* object so
        callers can mutate it without affecting the original."""
        from backend.blocks.agent import AgentExecutorBlock

        block = AgentExecutorBlock()
        input_data = {
            "user_id": "user-42",
            "graph_id": "graph-99",
            "graph_version": 3,
            "inputs": {"text": "hello"},
            "input_schema": {"props": "a"},
            "output_schema": {"props": "b"},
        }
        result = prepare_dry_run(block, input_data)

        assert result is not None
        # Must be a different object (copy, not alias)
        assert result is not input_data
        # Every key/value must be identical -- no modifications
        assert result == input_data
        # Mutating the copy must not affect the original
        result["extra"] = "added"
        assert "extra" not in input_data

    def test_regular_block_returns_none(self) -> None:
        block = _make_block()
        result = prepare_dry_run(block, {"query": "test"})
        assert result is None


# ---------------------------------------------------------------------------
# simulate_block – input/output passthrough
# ---------------------------------------------------------------------------


class TestSimulateBlockPassthrough:
    @pytest.mark.asyncio
    async def test_input_block_passthrough_with_value(self) -> None:
        from backend.blocks.io import AgentInputBlock

        block = AgentInputBlock()

        outputs = []
        async for name, data in simulate_block(block, {"value": "hello world"}):
            outputs.append((name, data))

        assert outputs == [("result", "hello world")]

    @pytest.mark.asyncio
    async def test_input_block_passthrough_without_value_uses_name(self) -> None:
        from backend.blocks.io import AgentInputBlock

        block = AgentInputBlock()

        outputs = []
        async for name, data in simulate_block(block, {"name": "user_query"}):
            outputs.append((name, data))

        assert outputs == [("result", "user_query")]

    @pytest.mark.asyncio
    async def test_input_block_passthrough_uses_placeholder(self) -> None:
        from backend.blocks.io import AgentInputBlock

        block = AgentInputBlock()

        outputs = []
        async for name, data in simulate_block(
            block, {"placeholder_values": ["option1", "option2"]}
        ):
            outputs.append((name, data))

        assert outputs == [("result", "option1")]

    @pytest.mark.asyncio
    async def test_output_block_passthrough_no_format(self) -> None:
        from backend.blocks.io import AgentOutputBlock

        block = AgentOutputBlock()

        outputs = []
        async for name, data in simulate_block(
            block, {"value": "result data", "name": "output_name"}
        ):
            outputs.append((name, data))

        assert ("output", "result data") in outputs
        assert ("name", "output_name") in outputs

    @pytest.mark.asyncio
    async def test_output_block_with_format_applies_jinja2(self) -> None:
        """When a format string is provided, AgentOutputBlock simulation should
        apply Jinja2 formatting and yield only 'output' (no 'name' pin)."""
        from backend.blocks.io import AgentOutputBlock

        block = AgentOutputBlock()

        outputs = []
        async for name, data in simulate_block(
            block,
            {
                "value": "Hello, World!",
                "name": "output_1",
                "format": "{{ output_1 }}!!",
            },
        ):
            outputs.append((name, data))

        assert len(outputs) == 1
        assert outputs[0] == ("output", "Hello, World!!!")

    @pytest.mark.asyncio
    async def test_output_block_with_format_no_name_pin(self) -> None:
        """When format is provided, the 'name' pin must NOT be yielded."""
        from backend.blocks.io import AgentOutputBlock

        block = AgentOutputBlock()

        output_names = []
        async for name, data in simulate_block(
            block,
            {
                "value": "42",
                "name": "output_2",
                "format": "{{ output_2 }}",
            },
        ):
            output_names.append(name)

        assert "name" not in output_names

    @pytest.mark.asyncio
    async def test_input_block_no_value_no_name_empty_placeholders(self) -> None:
        """AgentInputBlock with value=None, name=None, and empty
        placeholder_values list must not crash.

        When the ``name`` key is present but explicitly ``None``,
        ``dict.get("name", "sample input")`` returns ``None`` (the key
        exists), so the fallback sentinel is *not* used.  The test verifies
        the code does not raise and yields a single result."""
        from backend.blocks.io import AgentInputBlock

        block = AgentInputBlock()

        outputs = []
        async for name, data in simulate_block(
            block, {"value": None, "name": None, "placeholder_values": []}
        ):
            outputs.append((name, data))

        # Does not crash; yields exactly one output
        assert len(outputs) == 1
        assert outputs[0][0] == "result"

    @pytest.mark.asyncio
    async def test_input_block_missing_all_fields_uses_sentinel(self) -> None:
        """AgentInputBlock with no value, name, or placeholders at all should
        fall back to the ``"sample input"`` sentinel."""
        from backend.blocks.io import AgentInputBlock

        block = AgentInputBlock()

        outputs = []
        async for name, data in simulate_block(block, {}):
            outputs.append((name, data))

        assert outputs == [("result", "sample input")]

    @pytest.mark.asyncio
    async def test_generic_block_zero_outputs_handled(self) -> None:
        """When the LLM returns a valid JSON object but none of the output pins
        have meaningful values, ``simulate_block`` should yield nothing (zero
        outputs) and *not* raise or crash."""
        block = _make_block()

        with patch(
            "backend.executor.simulator._call_llm_for_simulation",
            new_callable=AsyncMock,
            # All output pin values are None or empty -- nothing to yield
            return_value={"result": None, "error": ""},
        ):
            outputs = []
            async for name, data in simulate_block(block, {"query": "test"}):
                outputs.append((name, data))

            assert outputs == []

    @pytest.mark.asyncio
    async def test_generic_block_calls_llm(self) -> None:
        """Generic blocks should call _call_llm_for_simulation."""
        block = _make_block()

        with patch(
            "backend.executor.simulator._call_llm_for_simulation",
            new_callable=AsyncMock,
            return_value={"result": "simulated result", "error": ""},
        ) as mock_llm:
            outputs = []
            async for name, data in simulate_block(block, {"query": "test"}):
                outputs.append((name, data))

            mock_llm.assert_called_once()
            assert ("result", "simulated result") in outputs
            # Empty error pin is omitted — not yielded
            assert ("error", "") not in outputs

    @pytest.mark.asyncio
    async def test_generic_block_omits_missing_pins(self) -> None:
        """Missing output pins are omitted (not yielded)."""
        block = _make_block()

        with patch(
            "backend.executor.simulator._call_llm_for_simulation",
            new_callable=AsyncMock,
            return_value={"result": "data"},  # missing "error" pin
        ):
            outputs: dict[str, Any] = {}
            async for name, data in simulate_block(block, {"query": "test"}):
                outputs[name] = data

            assert outputs["result"] == "data"
            # Missing pins are omitted — only meaningful values are yielded
            assert "error" not in outputs

    @pytest.mark.asyncio
    async def test_llm_failure_yields_error(self) -> None:
        """When LLM fails, should yield an error tuple."""
        block = _make_block()

        with patch(
            "backend.executor.simulator._call_llm_for_simulation",
            new_callable=AsyncMock,
            side_effect=RuntimeError("No client"),
        ):
            outputs = []
            async for name, data in simulate_block(block, {"query": "test"}):
                outputs.append((name, data))

            assert len(outputs) == 1
            assert outputs[0][0] == "error"
            assert "No client" in outputs[0][1]
