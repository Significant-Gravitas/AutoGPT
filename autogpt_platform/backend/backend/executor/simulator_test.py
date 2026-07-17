"""Tests for the LLM-powered block simulator (dry-run execution).

Covers:
  - Prompt building (credential stripping, realistic-output instructions)
  - Input/output block passthrough
  - prepare_dry_run routing
  - simulate_block output-pin filling
  - Default simulator model + OpenRouter cost tracking
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from backend.blocks.llm import LlmModel
from backend.blocks.orchestrator import ExecutionMode, OrchestratorBlock
from backend.executor.simulator import (
    _DEFAULT_SIMULATOR_MODEL,
    _extract_cost_usd,
    _truncate_input_values,
    _truncate_value,
    build_simulation_prompt,
    get_dry_run_credentials,
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
    def test_orchestrator_uses_simulation_model(self) -> None:
        """OrchestratorBlock should use the simulation model and cap iterations."""
        from unittest.mock import patch

        from backend.blocks.orchestrator import OrchestratorBlock

        block = OrchestratorBlock()
        with patch(
            "backend.executor.simulator._get_platform_openrouter_key",
            return_value="sk-or-test-key",
        ):
            result = prepare_dry_run(
                block,
                {"agent_mode_max_iterations": 10, "model": "gpt-4o", "other": "val"},
            )
        assert result is not None
        # Capped to min(original, 10) — user's 10 passes through unchanged.
        assert result["agent_mode_max_iterations"] == 10
        assert result["other"] == "val"
        assert result["model"] != "gpt-4o"  # overridden to simulation model
        # Simulation model must parse as a real LlmModel so OrchestratorBlock's
        # Pydantic input validation accepts it.
        assert LlmModel(result["model"]) is not None
        # The injected model must be a canonical LlmModel value (string equal
        # to one of the enum's ``.value``s), not an OpenRouter alias slug —
        # OrchestratorBlock.validate_data → jsonschema only accepts literal
        # ``LlmModel.value``s in the schema's ``enum``, and the alias map
        # in ``LlmModel._missing_`` does not surface in the generated
        # JSON Schema.  Anything else trips
        # ``"'<slug>' is not one of [...]"`` at runtime.
        canonical_values = {m.value for m in LlmModel}
        assert result["model"] in canonical_values, (
            f"prepare_dry_run injected non-canonical model {result['model']!r}; "
            f"jsonschema validation will reject it"
        )
        # credentials left as-is so block schema validation passes —
        # actual creds injected via extra_exec_kwargs in manager.py
        assert "credentials" not in result
        assert result["_dry_run_api_key"] == "sk-or-test-key"

    def test_orchestrator_invalid_sim_model_override_falls_back_to_default(
        self,
    ) -> None:
        """An invalid ``CHAT_SIMULATION_MODEL`` env value must not crash
        ``prepare_dry_run`` — fall back to the default so dry-run keeps
        working.  Without the guard, ``LlmModel('<garbage>')`` raises
        ``ValueError`` and aborts every Orchestrator dry-run."""
        with (
            patch(
                "backend.executor.simulator._get_platform_openrouter_key",
                return_value="sk-or-test-key",
            ),
            patch(
                "backend.executor.simulator._simulator_model",
                return_value="not-a-real-model-slug",
            ),
        ):
            result = prepare_dry_run(
                OrchestratorBlock(),
                {
                    "prompt": "test",
                    "model": LlmModel.CLAUDE_4_7_OPUS.value,
                    "agent_mode_max_iterations": 1,
                },
            )
        assert result is not None
        # Must land on the default value, not the garbage override.
        assert result["model"] == LlmModel(_DEFAULT_SIMULATOR_MODEL).value, (
            "Invalid CHAT_SIMULATION_MODEL should fall back to default; "
            f"got {result['model']!r}"
        )

    def test_orchestrator_forces_built_in_execution_mode(self) -> None:
        """prepare_dry_run overrides ``execution_mode`` to ``BUILT_IN``
        regardless of user choice.  With ``sim_model`` defaulting to
        Gemini Flash-Lite (provider=open_router):

          - BUILT_IN routes ``llm.llm_call`` to its open_router branch
            (OpenAI SDK against openrouter.ai with the OR key) — works.
          - EXTENDED_THINKING would hit the SDK subprocess's
            ``model.value.startswith("claude")`` guard and raise
            ``ValueError`` for any non-Claude sim_model.

        Honouring the user's pick would force sim_model back to Claude
        (to satisfy EXTENDED_THINKING), which in turn breaks the
        LLM-simulation path for every non-Orchestrator block in the
        same graph (Claude wraps JSON-mode output in markdown fences,
        Gemini doesn't)."""
        block = OrchestratorBlock()
        with patch(
            "backend.executor.simulator._get_platform_openrouter_key",
            return_value="sk-or-test-key",
        ):
            # User explicitly picked EXTENDED_THINKING — the dry-run
            # still overrides to BUILT_IN.
            result = prepare_dry_run(
                block,
                {
                    "prompt": "test",
                    "model": LlmModel.CLAUDE_4_7_OPUS.value,
                    "execution_mode": ExecutionMode.EXTENDED_THINKING.value,
                    "agent_mode_max_iterations": 1,
                },
            )
        assert result is not None
        assert result["execution_mode"] == ExecutionMode.BUILT_IN.value, (
            f"prepare_dry_run must force BUILT_IN to keep Gemini sim_model "
            f"off the SDK's Claude-only gate; got {result['execution_mode']!r}"
        )

    def test_orchestrator_input_passes_jsonschema_validation(self) -> None:
        """The injected dry-run input must pass OrchestratorBlock.validate_data.

        Pinning this prevents the SECRT-2368 follow-up bug class where
        prepare_dry_run injects an OpenRouter slug that LlmModel resolves
        via the alias map at the Pydantic layer, but jsonschema enum
        validation (which runs *before* Pydantic) rejects.
        """
        block = OrchestratorBlock()
        user_input = {
            "prompt": "test",
            "model": LlmModel.CLAUDE_4_7_OPUS.value,
            "credentials": {
                "id": "00000000-0000-0000-0000-000000000000",
                "provider": "open_router",
                "type": "api_key",
            },
            "agent_mode_max_iterations": 1,
            "execution_mode": "built_in",
            "multiple_tool_calls": False,
            "max_tokens": 50,
            "retry": 0,
        }
        with patch(
            "backend.executor.simulator._get_platform_openrouter_key",
            return_value="sk-or-test-key",
        ):
            dry_input = prepare_dry_run(block, user_input)
        assert dry_input is not None
        # Strip simulator-internal markers before validating, just like
        # manager.py does before calling Input(**...).
        validation_input = {k: v for k, v in dry_input.items() if not k.startswith("_")}
        err = block.input_schema.validate_data(validation_input)
        assert (
            err is None
        ), f"prepare_dry_run produced input that fails jsonschema validation: {err}"

    def test_orchestrator_zero_stays_zero(self) -> None:
        from unittest.mock import patch

        from backend.blocks.orchestrator import OrchestratorBlock

        block = OrchestratorBlock()
        with patch(
            "backend.executor.simulator._get_platform_openrouter_key",
            return_value="sk-or-test-key",
        ):
            result = prepare_dry_run(block, {"agent_mode_max_iterations": 0})
        assert result is not None
        assert result["agent_mode_max_iterations"] == 0

    def test_orchestrator_falls_back_without_key(self) -> None:
        """Without platform OpenRouter key, OrchestratorBlock falls back
        to LLM simulation (returns None)."""
        from unittest.mock import patch

        from backend.blocks.orchestrator import OrchestratorBlock

        block = OrchestratorBlock()
        with patch(
            "backend.executor.simulator._get_platform_openrouter_key",
            return_value=None,
        ):
            result = prepare_dry_run(block, {"agent_mode_max_iterations": 5})
        assert result is None

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


class TestGetDryRunCredentials:
    """get_dry_run_credentials pops _dry_run_api_key and returns APIKeyCredentials.

    The returned object must have fields that can be serialised into a valid
    CredentialsMetaInput placeholder dict for manager.py's schema-construction fix
    (Bug: manager.py nullified input_data[field_name] = None, which caused
    _execute's input_schema(**...) to fail because required credential fields were
    missing after the None-filter pass).
    """

    def test_returns_credentials_when_key_present(self) -> None:
        input_data = {"_dry_run_api_key": "sk-or-test", "other": "val"}
        creds = get_dry_run_credentials(input_data)
        assert creds is not None
        assert creds.api_key.get_secret_value() == "sk-or-test"
        # key is consumed from input_data
        assert "_dry_run_api_key" not in input_data

    def test_returns_none_when_key_absent(self) -> None:
        input_data: dict = {"other": "val"}
        creds = get_dry_run_credentials(input_data)
        assert creds is None

    def test_credentials_have_metadata_fields_for_placeholder(self) -> None:
        """The returned credentials must have id, provider, type, and title so
        manager.py can synthesise a valid CredentialsMetaInput placeholder."""
        from backend.integrations.providers import ProviderName

        creds = get_dry_run_credentials({"_dry_run_api_key": "sk-or-test"})
        assert creds is not None
        assert creds.id == "dry-run-platform"
        assert creds.provider == ProviderName.OPEN_ROUTER
        assert creds.type == "api_key"
        assert creds.title is not None


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
            block, {"options": ["option1", "option2"]}
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
    async def test_input_block_no_value_no_name_empty_options(self) -> None:
        """AgentInputBlock with value=None, name=None, and empty
        options list must not crash.

        When the ``name`` key is present but explicitly ``None``,
        ``dict.get("name", "sample input")`` returns ``None`` (the key
        exists), so the fallback sentinel is *not* used.  The test verifies
        the code does not raise and yields a single result."""
        from backend.blocks.io import AgentInputBlock

        block = AgentInputBlock()

        outputs = []
        async for name, data in simulate_block(
            block, {"value": None, "name": None, "options": []}
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
        have meaningful values, ``simulate_block`` should still yield defaults
        for required output pins so downstream nodes don't stall."""
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

            # "result" is required, so a default empty string is yielded
            assert outputs == [("result", "")]

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
    async def test_generic_block_preserves_falsy_values(self) -> None:
        """Valid falsy values like False, 0, and [] must be yielded, not dropped."""
        block = _make_block(
            output_schema={
                "properties": {
                    "flag": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "items": {"type": "array"},
                },
                "required": ["flag", "count", "items"],
            }
        )

        with patch(
            "backend.executor.simulator._call_llm_for_simulation",
            new_callable=AsyncMock,
            return_value={"flag": False, "count": 0, "items": []},
        ):
            outputs: dict[str, Any] = {}
            async for name, data in simulate_block(block, {"query": "test"}):
                outputs[name] = data

            assert outputs["flag"] is False
            assert outputs["count"] == 0
            assert outputs["items"] == []

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


# ---------------------------------------------------------------------------
# Default model + OpenRouter cost tracking
# ---------------------------------------------------------------------------


def _sim_usage(
    *,
    prompt_tokens: int = 1200,
    completion_tokens: int = 300,
    cost: object = 0.000157,
) -> CompletionUsage:
    """Typed ``CompletionUsage`` carrying OpenRouter's ``cost`` extension
    via ``model_extra`` — same pattern as
    ``copilot/tools/web_search_test.py::_usage``.  ``model_construct``
    preserves unknown fields; ``model_validate`` would drop them."""
    payload: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cost is not None:
        payload["cost"] = cost
    return CompletionUsage.model_construct(None, **payload)


def _sim_completion(*, content: str, usage: CompletionUsage) -> ChatCompletion:
    """Typed ``ChatCompletion`` shaped like an OpenRouter simulator
    response so the production code runs under real SDK types."""
    message = ChatCompletionMessage.model_construct(
        None, role="assistant", content=content
    )
    choice = Choice.model_construct(
        None, index=0, finish_reason="stop", message=message
    )
    return ChatCompletion.model_construct(
        None,
        id="cmpl-sim",
        object="chat.completion",
        created=0,
        model=_DEFAULT_SIMULATOR_MODEL,
        choices=[choice],
        usage=usage,
    )


class TestDefaultSimulatorModel:
    """Pin the default model.  Four guards line up with the constraints
    laid out next to ``_DEFAULT_SIMULATOR_MODEL`` in ``simulator.py``:
    value pin, ``LlmModel`` parseability, OpenRouter slug shape, and
    ``open_router`` provider routing (so the BUILT_IN orchestrator path
    in ``llm.llm_call`` doesn't get routed through ``api.anthropic.com``
    with the platform OR key)."""

    def test_default_is_gemini_flash_lite(self) -> None:
        assert _DEFAULT_SIMULATOR_MODEL == "google/gemini-2.5-flash-lite"

    def test_default_parses_as_llm_model(self) -> None:
        assert LlmModel(_DEFAULT_SIMULATOR_MODEL) is LlmModel.GEMINI_2_5_FLASH_LITE

    def test_default_is_openrouter_slug(self) -> None:
        # The LLM-simulation path hits OpenRouter's OpenAI-compat endpoint,
        # which only accepts canonical ``<vendor>/<model>`` slugs.
        assert "/" in _DEFAULT_SIMULATOR_MODEL

    def test_default_provider_is_open_router(self) -> None:
        # ``llm.llm_call`` dispatches on ``llm_model.metadata.provider``.
        # An ``anthropic`` provider would route OR-dry-run-credentials
        # at ``api.anthropic.com`` → 401.  Pin ``open_router`` here so
        # a future default change that breaks this routing trips at
        # unit-test time.
        assert LlmModel(_DEFAULT_SIMULATOR_MODEL).metadata.provider == "open_router"


class TestExtractCostUsd:
    """Provider-reported USD cost via typed ``model_extra`` — mirrors
    ``copilot.tools.web_search._extract_cost_usd`` and
    ``copilot.baseline.service._extract_usage_cost``."""

    def test_returns_cost_value(self) -> None:
        assert _extract_cost_usd(_sim_usage(cost=0.000157)) == pytest.approx(0.000157)

    def test_returns_none_when_usage_missing(self) -> None:
        assert _extract_cost_usd(None) is None

    def test_returns_none_when_cost_field_missing(self) -> None:
        assert _extract_cost_usd(_sim_usage(cost=None)) is None

    def test_returns_none_when_cost_is_explicit_null(self) -> None:
        usage = CompletionUsage.model_construct(
            None, prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=None
        )
        assert _extract_cost_usd(usage) is None

    def test_returns_none_when_cost_is_negative(self) -> None:
        usage = CompletionUsage.model_construct(
            None, prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=-0.5
        )
        assert _extract_cost_usd(usage) is None

    def test_accepts_numeric_string(self) -> None:
        usage = CompletionUsage.model_construct(
            None, prompt_tokens=0, completion_tokens=0, total_tokens=0, cost="0.017"
        )
        assert _extract_cost_usd(usage) == pytest.approx(0.017)


class TestSimulatorCostTracking:
    """Integration: mock the OpenAI client, confirm the simulator sends
    the flash-lite default + extra_body, then plumbs through to
    ``persist_and_record_usage`` with ``provider='open_router'`` and the
    real ``usage.cost`` pulled off ``model_extra``."""

    def _mock_client(self, fake_resp: ChatCompletion) -> tuple[Any, AsyncMock]:
        """Build a fake ``AsyncOpenAI`` client.  Same nested-type pattern as
        ``copilot/tools/web_search_test.py::_mock_client`` — avoids
        MagicMock's auto-child-attr behaviour so the exact ``create`` call
        surface is what gets invoked."""
        create_mock = AsyncMock(return_value=fake_resp)
        client = type(
            "MC",
            (),
            {
                "chat": type(
                    "C",
                    (),
                    {"completions": type("CC", (), {"create": create_mock})()},
                )()
            },
        )()
        return client, create_mock

    @pytest.mark.asyncio
    async def test_passes_default_model_and_tracks_cost(self) -> None:
        block = _make_block()
        fake_resp = _sim_completion(
            content='{"result": "simulated"}',
            usage=_sim_usage(prompt_tokens=1100, completion_tokens=220, cost=0.000189),
        )
        client, create_mock = self._mock_client(fake_resp)

        with (
            patch(
                "backend.executor.simulator.get_openai_client",
                return_value=client,
            ),
            patch(
                "backend.executor.simulator.persist_and_record_usage",
                new=AsyncMock(return_value=1320),
            ) as mock_track,
        ):
            outputs = []
            async for name, data in simulate_block(
                block, {"query": "hello"}, user_id="user-42"
            ):
                outputs.append((name, data))

        assert ("result", "simulated") in outputs

        create_kwargs = create_mock.await_args.kwargs
        assert create_kwargs["model"] == _DEFAULT_SIMULATOR_MODEL
        assert create_kwargs["extra_body"] == {"usage": {"include": True}}

        track_kwargs = mock_track.await_args.kwargs
        # The simulator routes through ``get_openai_client(prefer_openrouter=True)``,
        # which only ever hits OpenRouter (or None) under non-local transport — so
        # the cost row is always ``open_router`` here, never the chat transport's
        # identity. See ``clients_test.TestOpenrouterHelperCostProvider`` for the
        # per-transport matrix (incl. the subscription / direct_anthropic regression).
        assert track_kwargs["provider"] == "open_router"
        assert track_kwargs["model"] == _DEFAULT_SIMULATOR_MODEL
        assert track_kwargs["user_id"] == "user-42"
        assert track_kwargs["prompt_tokens"] == 1100
        assert track_kwargs["completion_tokens"] == 220
        assert track_kwargs["cost_usd"] == pytest.approx(0.000189)
        assert track_kwargs["session"] is None
        assert track_kwargs["log_prefix"] == "[simulator]"

    @pytest.mark.asyncio
    async def test_tracks_even_when_cost_absent(self) -> None:
        """Provider may omit ``cost`` (e.g. non-OpenRouter proxies).  We
        still record token counts — ``persist_and_record_usage`` logs the
        turn and skips the rate-limit ledger when cost is ``None``."""
        block = _make_block()
        fake_resp = _sim_completion(
            content='{"result": "ok"}',
            usage=_sim_usage(prompt_tokens=100, completion_tokens=20, cost=None),
        )
        client, _ = self._mock_client(fake_resp)

        with (
            patch(
                "backend.executor.simulator.get_openai_client",
                return_value=client,
            ),
            patch(
                "backend.executor.simulator.persist_and_record_usage",
                new=AsyncMock(return_value=120),
            ) as mock_track,
        ):
            async for _name, _data in simulate_block(
                block, {"query": "x"}, user_id="user-7"
            ):
                pass

        track_kwargs = mock_track.await_args.kwargs
        assert track_kwargs["cost_usd"] is None
        assert track_kwargs["user_id"] == "user-7"
        # Non-local prefer_openrouter route → always logged as ``open_router``.
        assert track_kwargs["provider"] == "open_router"

    @pytest.mark.asyncio
    async def test_tracking_failure_does_not_break_simulation(self) -> None:
        """Cost-tracking failures are warnings, not simulation failures —
        the block output must still flow to the caller."""
        block = _make_block()
        fake_resp = _sim_completion(
            content='{"result": "simulated"}',
            usage=_sim_usage(),
        )
        client, _ = self._mock_client(fake_resp)

        with (
            patch(
                "backend.executor.simulator.get_openai_client",
                return_value=client,
            ),
            patch(
                "backend.executor.simulator.persist_and_record_usage",
                new=AsyncMock(side_effect=RuntimeError("redis down")),
            ),
        ):
            outputs = []
            async for name, data in simulate_block(
                block, {"query": "hello"}, user_id="user-42"
            ):
                outputs.append((name, data))

        assert ("result", "simulated") in outputs
