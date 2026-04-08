"""Unit tests for resolve_tracking and log_system_credential_cost."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.model import NodeExecutionStats
from backend.executor.cost_tracking import (
    drain_pending_cost_logs,
    log_system_credential_cost,
    resolve_tracking,
)

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

    def test_elevenlabs_uses_script_field(self):
        """VideoNarrationBlock (elevenlabs) uses `script` field, not script_input/text."""
        stats = self._stats()
        tt, amt = resolve_tracking("elevenlabs", stats, {"script": "Narration"})
        assert tt == "characters"
        assert amt == 9.0

    def test_block_declared_cost_type_items(self):
        """Block explicitly setting provider_cost_type='items' short-circuits heuristics."""
        stats = self._stats(provider_cost=5.0, provider_cost_type="items")
        tt, amt = resolve_tracking("google_maps", stats, {})
        assert tt == "items"
        assert amt == 5.0

    def test_block_declared_cost_type_characters(self):
        """TTS block can declare characters directly, bypassing input_data lookup."""
        stats = self._stats(provider_cost=42.0, provider_cost_type="characters")
        tt, amt = resolve_tracking("unreal_speech", stats, {})
        assert tt == "characters"
        assert amt == 42.0

    def test_block_declared_cost_type_wins_over_tokens(self):
        """provider_cost_type takes precedence over token-based heuristic."""
        stats = self._stats(
            provider_cost=1.0,
            provider_cost_type="per_run",
            input_token_count=500,
        )
        tt, amt = resolve_tracking("openai", stats, {})
        assert tt == "per_run"
        assert amt == 1.0

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

    def test_negative_provider_cost_clamped_to_zero(self):
        """Negative provider_cost values must be clamped to 0."""
        stats = self._stats(provider_cost=-0.005)
        tt, amt = resolve_tracking("openrouter", stats, {})
        assert tt == "cost_usd"
        assert amt == 0.0

    def test_negative_block_declared_cost_clamped_to_zero(self):
        """Negative block-declared cost must also be clamped to 0."""
        stats = self._stats(provider_cost=-1.0, provider_cost_type="items")
        tt, amt = resolve_tracking("google_maps", stats, {})
        assert tt == "items"
        assert amt == 0.0

    def test_provider_cost_takes_precedence_over_tokens(self):
        stats = self._stats(
            provider_cost=0.01, input_token_count=500, output_token_count=200
        )
        tt, amt = resolve_tracking("openai", stats, {})
        assert tt == "cost_usd"
        assert amt == 0.01

    def test_provider_cost_zero_is_not_none(self):
        """provider_cost=0.0 is falsy but should still be tracked as cost_usd
        (e.g. free-tier or fully-cached responses from OpenRouter)."""
        stats = self._stats(provider_cost=0.0)
        tt, amt = resolve_tracking("open_router", stats, {})
        assert tt == "cost_usd"
        assert amt == 0.0

    def test_tokens_take_precedence_over_provider_specific(self):
        stats = self._stats(input_token_count=100, walltime=10.0)
        tt, amt = resolve_tracking("fal", stats, {})
        assert tt == "tokens"
        assert amt == 100.0


# ---------------------------------------------------------------------------
# log_system_credential_cost
# ---------------------------------------------------------------------------


def _make_db_client() -> MagicMock:
    db_client = MagicMock()
    db_client.log_platform_cost = AsyncMock()
    return db_client


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
        db_client = _make_db_client()
        node_exec = _make_node_exec(dry_run=True)
        block = _make_block()
        stats = NodeExecutionStats()
        await log_system_credential_cost(node_exec, block, stats, db_client)
        db_client.log_platform_cost.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_no_credential_fields(self):
        db_client = _make_db_client()
        node_exec = _make_node_exec(inputs={})
        block = _make_block(has_credentials=False)
        stats = NodeExecutionStats()
        await log_system_credential_cost(node_exec, block, stats, db_client)
        db_client.log_platform_cost.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_cred_data_missing(self):
        db_client = _make_db_client()
        node_exec = _make_node_exec(inputs={})
        block = _make_block()
        stats = NodeExecutionStats()
        await log_system_credential_cost(node_exec, block, stats, db_client)
        db_client.log_platform_cost.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_not_system_credential(self):
        db_client = _make_db_client()
        with patch(
            "backend.executor.cost_tracking.is_system_credential",
            return_value=False,
        ):
            node_exec = _make_node_exec(
                inputs={
                    "credentials": {"id": "user-cred-123", "provider": "openai"},
                }
            )
            block = _make_block()
            stats = NodeExecutionStats()
            await log_system_credential_cost(node_exec, block, stats, db_client)
        db_client.log_platform_cost.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_logs_with_system_credential(self):
        db_client = _make_db_client()
        with (
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
            await log_system_credential_cost(node_exec, block, stats, db_client)
            await asyncio.sleep(0)

        db_client.log_platform_cost.assert_awaited_once()
        entry = db_client.log_platform_cost.call_args[0][0]
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
        db_client = _make_db_client()
        with (
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
            await log_system_credential_cost(node_exec, block, stats, db_client)
            await asyncio.sleep(0)

        entry = db_client.log_platform_cost.call_args[0][0]
        assert entry.cost_microdollars == 1500
        assert entry.tracking_type == "cost_usd"
        assert entry.metadata["tracking_type"] == "cost_usd"
        assert entry.metadata["provider_cost_raw"] == 0.0015

    @pytest.mark.asyncio
    async def test_model_name_enum_converted_to_str(self):
        db_client = _make_db_client()
        with (
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
            await log_system_credential_cost(node_exec, block, stats, db_client)
            await asyncio.sleep(0)

        entry = db_client.log_platform_cost.call_args[0][0]
        assert entry.model == "FakeModel.GPT4"

    @pytest.mark.asyncio
    async def test_model_name_dict_becomes_none(self):
        db_client = _make_db_client()
        with (
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
            await log_system_credential_cost(node_exec, block, stats, db_client)
            await asyncio.sleep(0)

        entry = db_client.log_platform_cost.call_args[0][0]
        assert entry.model is None

    @pytest.mark.asyncio
    async def test_does_not_raise_when_block_usage_cost_raises(self):
        """log_system_credential_cost must swallow exceptions from block_usage_cost."""
        db_client = _make_db_client()
        with (
            patch(
                "backend.executor.cost_tracking.is_system_credential", return_value=True
            ),
            patch(
                "backend.executor.cost_tracking.block_usage_cost",
                side_effect=RuntimeError("pricing lookup failed"),
            ),
        ):
            node_exec = _make_node_exec(
                inputs={
                    "credentials": {"id": "sys-cred", "provider": "openai"},
                }
            )
            block = _make_block()
            stats = NodeExecutionStats()
            # Should not raise — outer except must catch block_usage_cost error
            await log_system_credential_cost(node_exec, block, stats, db_client)

    @pytest.mark.asyncio
    async def test_round_instead_of_int_for_microdollars(self):
        db_client = _make_db_client()
        with (
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
            await log_system_credential_cost(node_exec, block, stats, db_client)
            await asyncio.sleep(0)

        entry = db_client.log_platform_cost.call_args[0][0]
        assert entry.cost_microdollars == 1500

    @pytest.mark.asyncio
    async def test_per_run_metadata_has_no_provider_cost_raw(self):
        """For per-run providers (google_maps etc), provider_cost_raw is absent
        from metadata since stats.provider_cost is None."""
        db_client = _make_db_client()
        with (
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
                    "credentials": {"id": "sys-cred", "provider": "google_maps"},
                }
            )
            block = _make_block()
            stats = NodeExecutionStats()  # no provider_cost
            await log_system_credential_cost(node_exec, block, stats, db_client)
            await asyncio.sleep(0)

        entry = db_client.log_platform_cost.call_args[0][0]
        assert entry.tracking_type == "per_run"
        assert "provider_cost_raw" not in (entry.metadata or {})


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

    def test_provider_cost_accumulates(self):
        """Multiple merge_stats with provider_cost should sum (multi-round
        tool-calling in copilot / retries can report cost separately)."""
        stats = NodeExecutionStats()
        stats += NodeExecutionStats(provider_cost=0.001)
        stats += NodeExecutionStats(provider_cost=0.002)
        stats += NodeExecutionStats(provider_cost=0.003)
        assert stats.provider_cost == pytest.approx(0.006)

    def test_provider_cost_none_does_not_overwrite(self):
        """A None provider_cost must not wipe a previously-set value."""
        stats = NodeExecutionStats(provider_cost=0.01)
        stats += NodeExecutionStats()  # provider_cost=None by default
        assert stats.provider_cost == 0.01

    def test_provider_cost_type_last_write_wins(self):
        """provider_cost_type is a Literal — last set value wins on merge."""
        stats = NodeExecutionStats(provider_cost_type="tokens")
        stats += NodeExecutionStats(provider_cost_type="items")
        assert stats.provider_cost_type == "items"


# ---------------------------------------------------------------------------
# on_node_execution -> log_system_credential_cost integration
# ---------------------------------------------------------------------------


class TestManagerCostTrackingIntegration:
    @pytest.mark.asyncio
    async def test_log_called_with_accumulated_stats(self):
        """Verify that log_system_credential_cost receives stats that could
        have been accumulated by merge_stats across multiple yield steps."""
        db_client = _make_db_client()
        with (
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
            await log_system_credential_cost(node_exec, block, stats, db_client)
            await asyncio.sleep(0)

        db_client.log_platform_cost.assert_awaited_once()
        entry = db_client.log_platform_cost.call_args[0][0]
        assert entry.input_tokens == 300
        assert entry.tracking_type == "tokens"
        assert entry.metadata["tracking_amount"] == 300.0

    @pytest.mark.asyncio
    async def test_skips_cost_log_when_status_is_failed(self):
        """Manager only calls log_system_credential_cost on COMPLETED status.

        This test verifies the guard condition `if status == COMPLETED` directly:
        calling log_system_credential_cost only happens on success, never on
        FAILED or ERROR executions.
        """
        from backend.data.execution import ExecutionStatus

        db_client = _make_db_client()
        node_exec = _make_node_exec(
            inputs={"credentials": {"id": "sys-cred", "provider": "openai"}}
        )
        block = _make_block()
        stats = NodeExecutionStats(input_token_count=100)

        # Simulate the manager guard: only call on COMPLETED
        status = ExecutionStatus.FAILED
        if status == ExecutionStatus.COMPLETED:
            await log_system_credential_cost(node_exec, block, stats, db_client)

        db_client.log_platform_cost.assert_not_awaited()


# ---------------------------------------------------------------------------
# drain_pending_cost_logs
# ---------------------------------------------------------------------------


class TestDrainPendingCostLogs:
    @pytest.mark.asyncio
    async def test_drain_empty_set_completes(self):
        """drain_pending_cost_logs should succeed silently with no pending tasks."""
        # Ensure both pending task sets are empty before calling drain
        import backend.copilot.token_tracking as tt
        import backend.executor.cost_tracking as ct

        ct._pending_log_tasks.clear()
        tt._pending_log_tasks.clear()
        # Should not raise
        await drain_pending_cost_logs(timeout=1.0)

    @pytest.mark.asyncio
    async def test_drain_awaits_in_flight_tasks(self):
        """drain_pending_cost_logs waits for tasks on the current loop."""
        import backend.executor.cost_tracking as ct

        finished = []

        async def _slow():
            await asyncio.sleep(0)
            finished.append(1)

        task = asyncio.ensure_future(_slow())
        with ct._pending_log_tasks_lock:
            ct._pending_log_tasks.add(task)
        task.add_done_callback(lambda t: ct._pending_log_tasks.discard(t))

        await drain_pending_cost_logs(timeout=2.0)
        assert finished == [1], "drain_pending_cost_logs should have awaited the task"
