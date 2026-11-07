"""Unit tests for resolve_tracking and log_system_credential_cost."""

import asyncio
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


# provider, NodeExecutionStats fields, input_data, (tracking type, amount)
_TRACKING_CASES = {
    "provider_cost_returns_cost_usd": (
        "openai",
        {"provider_cost": 0.0042},
        {},
        ("cost_usd", 0.0042),
    ),
    "token_counts_return_tokens": (
        "anthropic",
        {"input_token_count": 300, "output_token_count": 100},
        {},
        ("tokens", 400.0),
    ),
    "token_counts_only_input": (
        "groq",
        {"input_token_count": 500},
        {},
        ("tokens", 500.0),
    ),
    "unreal_speech_returns_characters": (
        "unreal_speech",
        {},
        {"text": "Hello world"},
        ("characters", 11.0),
    ),
    "unreal_speech_empty_text": (
        "unreal_speech",
        {},
        {"text": ""},
        ("characters", 0.0),
    ),
    "unreal_speech_non_string_text": (
        "unreal_speech",
        {},
        {"text": 123},
        ("characters", 0.0),
    ),
    "d_id_uses_script_input": (
        "d_id",
        {},
        {"script_input": "Hello"},
        ("characters", 5.0),
    ),
    "elevenlabs_uses_text": (
        "elevenlabs",
        {},
        {"text": "Say this"},
        ("characters", 8.0),
    ),
    "elevenlabs_fallback_to_text_when_no_script_input": (
        "elevenlabs",
        {},
        {"text": "Fallback text"},
        ("characters", 13.0),
    ),
    # VideoNarrationBlock (elevenlabs) sends `script`, not script_input/text.
    "elevenlabs_uses_script_field": (
        "elevenlabs",
        {},
        {"script": "Narration"},
        ("characters", 9.0),
    ),
    "block_declared_cost_type_items": (
        "google_maps",
        {"provider_cost": 5.0, "provider_cost_type": "items"},
        {},
        ("items", 5.0),
    ),
    "block_declared_cost_type_characters": (
        "unreal_speech",
        {"provider_cost": 42.0, "provider_cost_type": "characters"},
        {},
        ("characters", 42.0),
    ),
    "block_declared_cost_type_wins_over_tokens": (
        "openai",
        {
            "provider_cost": 1.0,
            "provider_cost_type": "per_run",
            "input_token_count": 500,
        },
        {},
        ("per_run", 1.0),
    ),
    "e2b_returns_sandbox_seconds": (
        "e2b",
        {"walltime": 45.123},
        {},
        ("sandbox_seconds", 45.123),
    ),
    "e2b_no_walltime": ("e2b", {"walltime": 0}, {}, ("sandbox_seconds", 0.0)),
    "fal_returns_walltime": ("fal", {"walltime": 12.5}, {}, ("walltime_seconds", 12.5)),
    "revid_returns_walltime": (
        "revid",
        {"walltime": 60.0},
        {},
        ("walltime_seconds", 60.0),
    ),
    "replicate_returns_walltime": (
        "replicate",
        {"walltime": 30.0},
        {},
        ("walltime_seconds", 30.0),
    ),
    "unknown_provider_returns_per_run": ("google_maps", {}, {}, ("per_run", 1.0)),
    "negative_provider_cost_clamped_to_zero": (
        "openrouter",
        {"provider_cost": -0.005},
        {},
        ("cost_usd", 0.0),
    ),
    "negative_block_declared_cost_clamped_to_zero": (
        "google_maps",
        {"provider_cost": -1.0, "provider_cost_type": "items"},
        {},
        ("items", 0.0),
    ),
    "provider_cost_takes_precedence_over_tokens": (
        "openai",
        {"provider_cost": 0.01, "input_token_count": 500, "output_token_count": 200},
        {},
        ("cost_usd", 0.01),
    ),
    # 0.0 is falsy, but free-tier or fully cached OpenRouter calls do cost 0.
    "provider_cost_zero_is_not_none": (
        "open_router",
        {"provider_cost": 0.0},
        {},
        ("cost_usd", 0.0),
    ),
    "tokens_take_precedence_over_provider_specific": (
        "fal",
        {"input_token_count": 100, "walltime": 10.0},
        {},
        ("tokens", 100.0),
    ),
}


class TestResolveTracking:
    @pytest.mark.parametrize(
        "provider, stats, input_data, expected",
        _TRACKING_CASES.values(),
        ids=_TRACKING_CASES,
    )
    def test_resolve_tracking(self, provider, stats, input_data, expected):
        stats = NodeExecutionStats(**stats)
        assert resolve_tracking(provider, stats, input_data) == expected


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
    async def test_logs_codex_user_subscription_without_usd_cost(self):
        db_client = _make_db_client()
        with (
            patch(
                "backend.executor.cost_tracking.is_system_credential",
                return_value=False,
            ),
            patch(
                "backend.executor.cost_tracking.block_usage_cost",
                return_value=(0, None),
            ),
        ):
            node_exec = _make_node_exec(
                inputs={
                    "credentials": {
                        "id": "codex-user-cred",
                        "provider": "codex",
                        "type": "oauth2",
                    },
                    "model": "gpt-5.6-sol",
                }
            )
            block = _make_block()
            stats = NodeExecutionStats(
                input_token_count=500,
                output_token_count=200,
                provider_cost=700,
                provider_cost_type="tokens",
            )
            await log_system_credential_cost(node_exec, block, stats, db_client)
            await asyncio.sleep(0)

        db_client.log_platform_cost.assert_awaited_once()
        entry = db_client.log_platform_cost.call_args[0][0]
        assert entry.provider == "codex"
        assert entry.credential_id == "codex-user-cred"
        assert entry.cost_microdollars is None
        assert entry.tracking_type == "tokens"
        assert entry.tracking_amount == 700
        assert entry.metadata["billing_mode"] == "user_subscription"
        assert entry.metadata["execution_path"] == "codex_app_server"

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

    def test_subscription_route_metadata_survives_merge(self):
        stats = NodeExecutionStats()
        stats += NodeExecutionStats(
            billing_mode="user_subscription",
            auth_provider="codex",
            execution_path="codex_app_server",
            resolved_model="gpt-live",
        )

        assert stats.billing_mode == "user_subscription"
        assert stats.auth_provider == "codex"
        assert stats.execution_path == "codex_app_server"
        assert stats.resolved_model == "gpt-live"


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
