"""Unit tests for resolve_tracking and log_system_credential_cost."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.model import NodeExecutionStats
from backend.executor.cost_tracking import log_system_credential_cost, resolve_tracking

# ---------------------------------------------------------------------------
# resolve_tracking
# ---------------------------------------------------------------------------


class TestResolveTracking:
    def _stats(self, **overrides: Any) -> NodeExecutionStats:
        return NodeExecutionStats(**overrides)

    def test_provider_cost_returns_cost_usd(self):
        stats = self._stats(provider_cost=0.0042)
        tt, amt = resolve_tracking("openai", stats, {})
        assert tt == "cost_usd"
        assert amt == 0.0042

    def test_token_counts_return_tokens(self):
        stats = self._stats(input_token_count=300, output_token_count=100)
        tt, amt = resolve_tracking("anthropic", stats, {})
        assert tt == "tokens"
        assert amt == 400.0

    def test_token_counts_only_input(self):
        stats = self._stats(input_token_count=500)
        tt, amt = resolve_tracking("groq", stats, {})
        assert tt == "tokens"
        assert amt == 500.0

    def test_unreal_speech_returns_characters(self):
        stats = self._stats()
        tt, amt = resolve_tracking("unreal_speech", stats, {"text": "Hello world"})
        assert tt == "characters"
        assert amt == 11.0

    def test_unreal_speech_empty_text(self):
        stats = self._stats()
        tt, amt = resolve_tracking("unreal_speech", stats, {"text": ""})
        assert tt == "characters"
        assert amt == 0.0

    def test_unreal_speech_non_string_text(self):
        stats = self._stats()
        tt, amt = resolve_tracking("unreal_speech", stats, {"text": 123})
        assert tt == "characters"
        assert amt == 0.0

    def test_d_id_uses_script_input(self):
        stats = self._stats()
        tt, amt = resolve_tracking("d_id", stats, {"script_input": "Hello"})
        assert tt == "characters"
        assert amt == 5.0

    def test_elevenlabs_uses_text(self):
        stats = self._stats()
        tt, amt = resolve_tracking("elevenlabs", stats, {"text": "Say this"})
        assert tt == "characters"
        assert amt == 8.0

    def test_elevenlabs_fallback_to_text_when_no_script_input(self):
        stats = self._stats()
        tt, amt = resolve_tracking("elevenlabs", stats, {"text": "Fallback text"})
        assert tt == "characters"
        assert amt == 13.0

    def test_e2b_returns_sandbox_seconds(self):
        stats = self._stats(walltime=45.123)
        tt, amt = resolve_tracking("e2b", stats, {})
        assert tt == "sandbox_seconds"
        assert amt == 45.123

    def test_e2b_no_walltime(self):
        stats = self._stats(walltime=0)
        tt, amt = resolve_tracking("e2b", stats, {})
        assert tt == "sandbox_seconds"
        assert amt == 0.0

    def test_fal_returns_walltime(self):
        stats = self._stats(walltime=12.5)
        tt, amt = resolve_tracking("fal", stats, {})
        assert tt == "walltime_seconds"
        assert amt == 12.5

    def test_revid_returns_walltime(self):
        stats = self._stats(walltime=60.0)
        tt, amt = resolve_tracking("revid", stats, {})
        assert tt == "walltime_seconds"
        assert amt == 60.0

    def test_replicate_returns_walltime(self):
        stats = self._stats(walltime=30.0)
        tt, amt = resolve_tracking("replicate", stats, {})
        assert tt == "walltime_seconds"
        assert amt == 30.0

    def test_unknown_provider_returns_per_run(self):
        stats = self._stats()
        tt, amt = resolve_tracking("google_maps", stats, {})
        assert tt == "per_run"
        assert amt == 1.0

    def test_provider_cost_takes_precedence_over_tokens(self):
        stats = self._stats(
            provider_cost=0.01, input_token_count=500, output_token_count=200
        )
        tt, amt = resolve_tracking("openai", stats, {})
        assert tt == "cost_usd"
        assert amt == 0.01

    def test_tokens_take_precedence_over_provider_specific(self):
        stats = self._stats(input_token_count=100, walltime=10.0)
        tt, amt = resolve_tracking("fal", stats, {})
        assert tt == "tokens"
        assert amt == 100.0


# ---------------------------------------------------------------------------
# log_system_credential_cost
# ---------------------------------------------------------------------------


def _make_block(has_credentials: bool = True) -> MagicMock:
    block = MagicMock()
    block.name = "TestBlock"
    input_schema = MagicMock()
    if has_credentials:
        input_schema.get_credentials_fields.return_value = {"credentials": MagicMock()}
    else:
        input_schema.get_credentials_fields.return_value = {}
    block.input_schema = input_schema
    return block


def _make_node_exec(
    inputs: dict | None = None,
    dry_run: bool = False,
) -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="gx-1",
        graph_id="g-1",
        graph_version=1,
        node_exec_id="nx-1",
        node_id="n-1",
        block_id="b-1",
        inputs=inputs or {},
        execution_context=ExecutionContext(dry_run=dry_run),
    )


class TestLogSystemCredentialCost:
    @pytest.mark.asyncio
    async def test_skips_dry_run(self):
        mock_log = AsyncMock()
        with patch(
            "backend.executor.cost_tracking.log_platform_cost_safe", new=mock_log
        ):
            node_exec = _make_node_exec(dry_run=True)
            block = _make_block()
            stats = NodeExecutionStats()
            await log_system_credential_cost(node_exec, block, stats)
        mock_log.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_no_credential_fields(self):
        mock_log = AsyncMock()
        with patch(
            "backend.executor.cost_tracking.log_platform_cost_safe", new=mock_log
        ):
            node_exec = _make_node_exec(inputs={})
            block = _make_block(has_credentials=False)
            stats = NodeExecutionStats()
            await log_system_credential_cost(node_exec, block, stats)
        mock_log.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_cred_data_missing(self):
        mock_log = AsyncMock()
        with patch(
            "backend.executor.cost_tracking.log_platform_cost_safe", new=mock_log
        ):
            node_exec = _make_node_exec(inputs={})
            block = _make_block()
            stats = NodeExecutionStats()
            await log_system_credential_cost(node_exec, block, stats)
        mock_log.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_not_system_credential(self):
        mock_log = AsyncMock()
        with (
            patch(
                "backend.executor.cost_tracking.log_platform_cost_safe", new=mock_log
            ),
            patch(
                "backend.executor.cost_tracking.is_system_credential",
                return_value=False,
            ),
        ):
            node_exec = _make_node_exec(
                inputs={
                    "credentials": {"id": "user-cred-123", "provider": "openai"},
                }
            )
            block = _make_block()
            stats = NodeExecutionStats()
            await log_system_credential_cost(node_exec, block, stats)
        mock_log.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_logs_with_system_credential(self):
        mock_log = AsyncMock()
        with (
            patch(
                "backend.executor.cost_tracking.log_platform_cost_safe", new=mock_log
            ),
            patch(
                "backend.executor.cost_tracking.is_system_credential", return_value=True
            ),
            patch(
                "backend.executor.cost_tracking.block_usage_cost",
                return_value=(10, None),
            ),
        ):
            node_exec = _make_node_exec(
                inputs={
                    "credentials": {"id": "sys-cred-1", "provider": "openai"},
                    "model": "gpt-4",
                }
            )
            block = _make_block()
            stats = NodeExecutionStats(input_token_count=500, output_token_count=200)
            await log_system_credential_cost(node_exec, block, stats)
            await asyncio.sleep(0)

        mock_log.assert_awaited_once()
        entry = mock_log.call_args[0][0]
        assert entry.user_id == "user-1"
        assert entry.provider == "openai"
        assert entry.block_name == "TestBlock"
        assert entry.model == "gpt-4"
        assert entry.input_tokens == 500
        assert entry.output_tokens == 200
        assert entry.tracking_type == "tokens"
        assert entry.metadata["tracking_type"] == "tokens"
        assert entry.metadata["tracking_amount"] == 700.0
        assert entry.metadata["credit_cost"] == 10

    @pytest.mark.asyncio
    async def test_logs_with_provider_cost(self):
        mock_log = AsyncMock()
        with (
            patch(
                "backend.executor.cost_tracking.log_platform_cost_safe", new=mock_log
            ),
            patch(
                "backend.executor.cost_tracking.is_system_credential", return_value=True
            ),
            patch(
                "backend.executor.cost_tracking.block_usage_cost",
                return_value=(5, None),
            ),
        ):
            node_exec = _make_node_exec(
                inputs={
                    "credentials": {"id": "sys-cred-2", "provider": "open_router"},
                }
            )
            block = _make_block()
            stats = NodeExecutionStats(provider_cost=0.0015)
            await log_system_credential_cost(node_exec, block, stats)
            await asyncio.sleep(0)

        entry = mock_log.call_args[0][0]
        assert entry.cost_microdollars == 1500
        assert entry.tracking_type == "cost_usd"
        assert entry.metadata["tracking_type"] == "cost_usd"
        assert entry.metadata["provider_cost_usd"] == 0.0015

    @pytest.mark.asyncio
    async def test_model_name_enum_converted_to_str(self):
        mock_log = AsyncMock()
        with (
            patch(
                "backend.executor.cost_tracking.log_platform_cost_safe", new=mock_log
            ),
            patch(
                "backend.executor.cost_tracking.is_system_credential", return_value=True
            ),
            patch(
                "backend.executor.cost_tracking.block_usage_cost",
                return_value=(0, None),
            ),
        ):
            from enum import Enum

            class FakeModel(Enum):
                GPT4 = "gpt-4"

            node_exec = _make_node_exec(
                inputs={
                    "credentials": {"id": "sys-cred", "provider": "openai"},
                    "model": FakeModel.GPT4,
                }
            )
            block = _make_block()
            stats = NodeExecutionStats()
            await log_system_credential_cost(node_exec, block, stats)
            await asyncio.sleep(0)

        entry = mock_log.call_args[0][0]
        assert entry.model == "FakeModel.GPT4"

    @pytest.mark.asyncio
    async def test_model_name_dict_becomes_none(self):
        mock_log = AsyncMock()
        with (
            patch(
                "backend.executor.cost_tracking.log_platform_cost_safe", new=mock_log
            ),
            patch(
                "backend.executor.cost_tracking.is_system_credential", return_value=True
            ),
            patch(
                "backend.executor.cost_tracking.block_usage_cost",
                return_value=(0, None),
            ),
        ):
            node_exec = _make_node_exec(
                inputs={
                    "credentials": {"id": "sys-cred", "provider": "openai"},
                    "model": {"nested": "value"},
                }
            )
            block = _make_block()
            stats = NodeExecutionStats()
            await log_system_credential_cost(node_exec, block, stats)
            await asyncio.sleep(0)

        entry = mock_log.call_args[0][0]
        assert entry.model is None

    @pytest.mark.asyncio
    async def test_round_instead_of_int_for_microdollars(self):
        mock_log = AsyncMock()
        with (
            patch(
                "backend.executor.cost_tracking.log_platform_cost_safe", new=mock_log
            ),
            patch(
                "backend.executor.cost_tracking.is_system_credential", return_value=True
            ),
            patch(
                "backend.executor.cost_tracking.block_usage_cost",
                return_value=(0, None),
            ),
        ):
            node_exec = _make_node_exec(
                inputs={
                    "credentials": {"id": "sys-cred", "provider": "openai"},
                }
            )
            block = _make_block()
            # 0.0015 * 1_000_000 = 1499.9999999... with float math
            # round() should give 1500, int() would give 1499
            stats = NodeExecutionStats(provider_cost=0.0015)
            await log_system_credential_cost(node_exec, block, stats)
            await asyncio.sleep(0)

        entry = mock_log.call_args[0][0]
        assert entry.cost_microdollars == 1500


# ---------------------------------------------------------------------------
# merge_stats accumulation
# ---------------------------------------------------------------------------


class TestMergeStats:
    """Tests for NodeExecutionStats accumulation via += (used by Block.merge_stats)."""

    def test_accumulates_output_size(self):
        stats = NodeExecutionStats()
        stats += NodeExecutionStats(output_size=10)
        stats += NodeExecutionStats(output_size=25)
        assert stats.output_size == 35

    def test_accumulates_tokens(self):
        stats = NodeExecutionStats()
        stats += NodeExecutionStats(input_token_count=100, output_token_count=50)
        stats += NodeExecutionStats(input_token_count=200, output_token_count=150)
        assert stats.input_token_count == 300
        assert stats.output_token_count == 200

    def test_preserves_provider_cost(self):
        stats = NodeExecutionStats()
        stats += NodeExecutionStats(provider_cost=0.005)
        stats += NodeExecutionStats(output_size=10)
        assert stats.provider_cost == 0.005
        assert stats.output_size == 10


# ---------------------------------------------------------------------------
# on_node_execution -> log_system_credential_cost integration
# ---------------------------------------------------------------------------


class TestManagerCostTrackingIntegration:
    @pytest.mark.asyncio
    async def test_log_called_with_accumulated_stats(self):
        """Verify that log_system_credential_cost receives stats that could
        have been accumulated by merge_stats across multiple yield steps."""
        mock_log = AsyncMock()
        with (
            patch(
                "backend.executor.cost_tracking.log_platform_cost_safe", new=mock_log
            ),
            patch(
                "backend.executor.cost_tracking.is_system_credential", return_value=True
            ),
            patch(
                "backend.executor.cost_tracking.block_usage_cost",
                return_value=(5, None),
            ),
        ):
            stats = NodeExecutionStats()
            stats += NodeExecutionStats(output_size=10, input_token_count=100)
            stats += NodeExecutionStats(output_size=25, input_token_count=200)

            assert stats.output_size == 35
            assert stats.input_token_count == 300

            node_exec = _make_node_exec(
                inputs={
                    "credentials": {"id": "sys-cred-acc", "provider": "openai"},
                    "model": "gpt-4",
                }
            )
            block = _make_block()
            await log_system_credential_cost(node_exec, block, stats)
            await asyncio.sleep(0)

        mock_log.assert_awaited_once()
        entry = mock_log.call_args[0][0]
        assert entry.input_tokens == 300
        assert entry.tracking_type == "tokens"
        assert entry.metadata["tracking_amount"] == 300.0
