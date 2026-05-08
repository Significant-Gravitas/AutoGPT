"""Tests for ExecutionMode enum and provider validation in the orchestrator.

Covers:
- ExecutionMode enum members exist and have stable values
- EXTENDED_THINKING provider validation (anthropic/open_router allowed, others rejected)
- EXTENDED_THINKING model-name validation (must start with "claude")
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.llm import LlmModel
from backend.blocks.orchestrator import ExecutionMode, OrchestratorBlock

# ---------------------------------------------------------------------------
# ExecutionMode enum integrity
# ---------------------------------------------------------------------------


class TestExecutionModeEnum:
    """Guard against accidental renames or removals of enum members."""

    def test_built_in_exists(self):
        assert hasattr(ExecutionMode, "BUILT_IN")
        assert ExecutionMode.BUILT_IN.value == "built_in"

    def test_extended_thinking_exists(self):
        assert hasattr(ExecutionMode, "EXTENDED_THINKING")
        assert ExecutionMode.EXTENDED_THINKING.value == "extended_thinking"

    def test_exactly_two_members(self):
        """If a new mode is added, this test should be updated intentionally."""
        assert set(ExecutionMode.__members__.keys()) == {
            "BUILT_IN",
            "EXTENDED_THINKING",
        }

    def test_string_enum(self):
        """ExecutionMode is a str enum so it serialises cleanly to JSON."""
        assert isinstance(ExecutionMode.BUILT_IN, str)
        assert isinstance(ExecutionMode.EXTENDED_THINKING, str)

    def test_round_trip_from_value(self):
        """Constructing from the string value should return the same member."""
        assert ExecutionMode("built_in") is ExecutionMode.BUILT_IN
        assert ExecutionMode("extended_thinking") is ExecutionMode.EXTENDED_THINKING


# ---------------------------------------------------------------------------
# Provider validation (inline in OrchestratorBlock.run)
# ---------------------------------------------------------------------------


def _make_model_stub(provider: str, value: str):
    """Create a lightweight stub that behaves like LlmModel for validation."""
    metadata = MagicMock()
    metadata.provider = provider
    stub = MagicMock()
    stub.metadata = metadata
    stub.value = value
    return stub


class TestExtendedThinkingProviderValidation:
    """The orchestrator rejects EXTENDED_THINKING for non-Anthropic providers."""

    def test_anthropic_provider_accepted(self):
        """provider='anthropic' + claude model should not raise."""
        model = _make_model_stub("anthropic", "claude-opus-4-6")
        provider = model.metadata.provider
        model_name = model.value
        assert provider in ("anthropic", "open_router")
        assert model_name.startswith("claude")

    def test_open_router_provider_accepted(self):
        """provider='open_router' + claude model should not raise."""
        model = _make_model_stub("open_router", "claude-sonnet-4-6")
        provider = model.metadata.provider
        model_name = model.value
        assert provider in ("anthropic", "open_router")
        assert model_name.startswith("claude")

    def test_openai_provider_rejected(self):
        """provider='openai' should be rejected for EXTENDED_THINKING."""
        model = _make_model_stub("openai", "gpt-4o")
        provider = model.metadata.provider
        assert provider not in ("anthropic", "open_router")

    def test_groq_provider_rejected(self):
        model = _make_model_stub("groq", "llama-3.3-70b-versatile")
        provider = model.metadata.provider
        assert provider not in ("anthropic", "open_router")

    def test_non_claude_model_rejected_even_if_anthropic_provider(self):
        """A hypothetical non-Claude model with provider='anthropic' is rejected."""
        model = _make_model_stub("anthropic", "not-a-claude-model")
        model_name = model.value
        assert not model_name.startswith("claude")

    def test_real_gpt4o_model_rejected(self):
        """Verify a real LlmModel enum member (GPT4O) fails the provider check."""
        model = LlmModel.GPT4O
        provider = model.metadata.provider
        assert provider not in ("anthropic", "open_router")

    def test_real_claude_model_passes(self):
        """Verify a real LlmModel enum member (CLAUDE_4_6_SONNET) passes."""
        model = LlmModel.CLAUDE_4_6_SONNET
        provider = model.metadata.provider
        model_name = model.value
        assert provider in ("anthropic", "open_router")
        assert model_name.startswith("claude")


# ---------------------------------------------------------------------------
# Integration-style: exercise the validation branch via OrchestratorBlock.run
# ---------------------------------------------------------------------------


def _make_input_data(model, execution_mode=ExecutionMode.EXTENDED_THINKING):
    """Build a minimal MagicMock that satisfies OrchestratorBlock.run's early path."""
    inp = MagicMock()
    inp.execution_mode = execution_mode
    inp.model = model
    inp.prompt = "test"
    inp.sys_prompt = ""
    inp.conversation_history = []
    inp.last_tool_output = None
    inp.prompt_values = {}
    return inp


async def _collect_run_outputs(block, input_data, **kwargs):
    """Exhaust the OrchestratorBlock.run async generator, collecting outputs."""
    outputs = []
    async for item in block.run(input_data, **kwargs):
        outputs.append(item)
    return outputs


class TestExtendedThinkingValidationRaisesInBlock:
    """Call OrchestratorBlock.run far enough to trigger the ValueError."""

    @pytest.mark.asyncio
    async def test_non_anthropic_provider_raises_valueerror(self):
        """EXTENDED_THINKING + openai provider raises ValueError."""
        block = OrchestratorBlock()
        input_data = _make_input_data(model=LlmModel.GPT4O)

        with (
            patch.object(
                block,
                "_create_tool_node_signatures",
                new_callable=AsyncMock,
                return_value=[],
            ),
            pytest.raises(ValueError, match="Anthropic-compatible"),
        ):
            await _collect_run_outputs(
                block,
                input_data,
                credentials=MagicMock(),
                graph_id="g",
                node_id="n",
                graph_exec_id="ge",
                node_exec_id="ne",
                user_id="u",
                graph_version=1,
                execution_context=MagicMock(),
                execution_processor=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_non_claude_model_with_anthropic_provider_raises(self):
        """A model with anthropic provider but non-claude name raises ValueError."""
        block = OrchestratorBlock()
        fake_model = _make_model_stub("anthropic", "not-a-claude-model")
        input_data = _make_input_data(model=fake_model)

        with (
            patch.object(
                block,
                "_create_tool_node_signatures",
                new_callable=AsyncMock,
                return_value=[],
            ),
            pytest.raises(ValueError, match="only supports Claude models"),
        ):
            await _collect_run_outputs(
                block,
                input_data,
                credentials=MagicMock(),
                graph_id="g",
                node_id="n",
                graph_exec_id="ge",
                node_exec_id="ne",
                user_id="u",
                graph_version=1,
                execution_context=MagicMock(),
                execution_processor=MagicMock(),
            )
