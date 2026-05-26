"""Unit tests for token_tracking.persist_and_record_usage.

Covers both the baseline (prompt+completion only) and SDK (with cache breakdown)
calling conventions, session persistence, and rate-limit recording.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from .model import ChatSession, Usage
from .token_tracking import persist_and_record_usage


def _make_session() -> ChatSession:
    """Return a minimal in-memory ChatSession for testing."""
    return ChatSession(
        session_id="sess-test",
        user_id="user-test",
        title=None,
        messages=[],
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Return value / total_tokens semantics
# ---------------------------------------------------------------------------


class TestTotalTokens:
    @pytest.mark.asyncio
    async def test_returns_prompt_plus_completion(self):
        """total_tokens = prompt + completion (cache excluded from total)."""
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new_callable=AsyncMock,
        ):
            total = await persist_and_record_usage(
                session=None,
                user_id=None,
                prompt_tokens=300,
                completion_tokens=200,
            )
        assert total == 500

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_tokens(self):
        """Returns 0 early when both prompt and completion are zero."""
        total = await persist_and_record_usage(
            session=None,
            user_id=None,
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert total == 0

    @pytest.mark.asyncio
    async def test_cache_tokens_excluded_from_total(self):
        """Cache tokens are stored separately and not added to total_tokens."""
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new_callable=AsyncMock,
        ):
            total = await persist_and_record_usage(
                session=None,
                user_id=None,
                prompt_tokens=100,
                completion_tokens=50,
                cache_read_tokens=5000,
                cache_creation_tokens=200,
            )
        # total = prompt + completion only (5000 + 200 cache excluded)
        assert total == 150

    @pytest.mark.asyncio
    async def test_baseline_path_no_cache(self):
        """Baseline (OpenRouter) path passes no cache tokens; total = prompt + completion."""
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new_callable=AsyncMock,
        ):
            total = await persist_and_record_usage(
                session=None,
                user_id="u1",
                prompt_tokens=1000,
                completion_tokens=400,
                log_prefix="[Baseline]",
            )
        assert total == 1400

    @pytest.mark.asyncio
    async def test_sdk_path_with_cache(self):
        """SDK (Anthropic) path passes cache tokens; total still = prompt + completion."""
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new_callable=AsyncMock,
        ):
            total = await persist_and_record_usage(
                session=None,
                user_id="u2",
                prompt_tokens=200,
                completion_tokens=100,
                cache_read_tokens=8000,
                cache_creation_tokens=400,
                log_prefix="[SDK]",
                cost_usd=0.0015,
            )
        assert total == 300


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    @pytest.mark.asyncio
    async def test_appends_usage_to_session(self):
        session = _make_session()
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new_callable=AsyncMock,
        ):
            await persist_and_record_usage(
                session=session,
                user_id=None,
                prompt_tokens=100,
                completion_tokens=50,
            )
        assert len(session.usage) == 1
        usage: Usage = session.usage[0]
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cache_read_tokens == 0
        assert usage.cache_creation_tokens == 0

    @pytest.mark.asyncio
    async def test_appends_cache_breakdown_to_session(self):
        session = _make_session()
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new_callable=AsyncMock,
        ):
            await persist_and_record_usage(
                session=session,
                user_id=None,
                prompt_tokens=200,
                completion_tokens=80,
                cache_read_tokens=3000,
                cache_creation_tokens=500,
            )
        usage: Usage = session.usage[0]
        assert usage.cache_read_tokens == 3000
        assert usage.cache_creation_tokens == 500

    @pytest.mark.asyncio
    async def test_multiple_turns_append_multiple_records(self):
        session = _make_session()
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new_callable=AsyncMock,
        ):
            await persist_and_record_usage(
                session=session, user_id=None, prompt_tokens=100, completion_tokens=50
            )
            await persist_and_record_usage(
                session=session, user_id=None, prompt_tokens=200, completion_tokens=70
            )
        assert len(session.usage) == 2

    @pytest.mark.asyncio
    async def test_none_session_does_not_raise(self):
        """When session is None (e.g. error path), no exception should be raised."""
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new_callable=AsyncMock,
        ):
            total = await persist_and_record_usage(
                session=None,
                user_id=None,
                prompt_tokens=100,
                completion_tokens=50,
            )
        assert total == 150

    @pytest.mark.asyncio
    async def test_no_append_when_zero_tokens(self):
        """When tokens are zero, function returns early — session unchanged."""
        session = _make_session()
        total = await persist_and_record_usage(
            session=session,
            user_id=None,
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert total == 0
        assert len(session.usage) == 0


# ---------------------------------------------------------------------------
# Rate-limit recording
# ---------------------------------------------------------------------------


class TestRateLimitRecording:
    @pytest.mark.asyncio
    async def test_calls_record_cost_usage_when_cost_and_user_id_present(self):
        """Rate-limit counter is charged with the real provider cost (microdollars)."""
        mock_record = AsyncMock()
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new=mock_record,
        ):
            await persist_and_record_usage(
                session=None,
                user_id="user-abc",
                prompt_tokens=100,
                completion_tokens=50,
                cache_read_tokens=1000,
                cache_creation_tokens=200,
                cost_usd=0.0123,
            )
        mock_record.assert_awaited_once_with(
            user_id="user-abc",
            cost_microdollars=12_300,
        )

    @pytest.mark.asyncio
    async def test_skips_record_when_cost_is_missing(self):
        """Without a provider cost we have no authoritative figure to charge."""
        mock_record = AsyncMock()
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new=mock_record,
        ):
            await persist_and_record_usage(
                session=None,
                user_id="user-abc",
                prompt_tokens=100,
                completion_tokens=50,
            )
        mock_record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_record_when_user_id_is_none(self):
        """Anonymous sessions should not create Redis keys."""
        mock_record = AsyncMock()
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new=mock_record,
        ):
            await persist_and_record_usage(
                session=None,
                user_id=None,
                prompt_tokens=100,
                completion_tokens=50,
                cost_usd=0.001,
            )
        mock_record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_record_usage_bubbles_unexpected_error(self):
        """Unexpected errors from record_cost_usage must propagate.

        record_cost_usage() owns its own (RedisError, ConnectionError, OSError)
        fail-open handling. Anything else is a real accounting bug and
        should not be silently swallowed at this layer.
        """
        mock_record = AsyncMock(side_effect=RuntimeError("boom"))
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new=mock_record,
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await persist_and_record_usage(
                    session=None,
                    user_id="user-xyz",
                    prompt_tokens=100,
                    completion_tokens=50,
                    cost_usd=0.002,
                )

    @pytest.mark.asyncio
    async def test_skips_record_when_zero_tokens_and_no_cost(self):
        """Returns 0 before calling record_cost_usage when there is nothing to record."""
        mock_record = AsyncMock()
        with patch(
            "backend.copilot.token_tracking.record_cost_usage",
            new=mock_record,
        ):
            await persist_and_record_usage(
                session=None,
                user_id="user-abc",
                prompt_tokens=0,
                completion_tokens=0,
            )
        mock_record.assert_not_awaited()


# ---------------------------------------------------------------------------
# PlatformCostLog integration
# ---------------------------------------------------------------------------


class TestPlatformCostLogging:
    @pytest.mark.asyncio
    async def test_logs_cost_entry_with_cost_usd(self):
        """When cost_usd is provided, tracking_type should be 'cost_usd'."""
        mock_log = AsyncMock()
        with (
            patch(
                "backend.copilot.token_tracking.record_cost_usage",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.copilot.token_tracking.platform_cost_db",
                return_value=type(
                    "FakePlatformCostDb", (), {"log_platform_cost": mock_log}
                )(),
            ),
        ):
            await persist_and_record_usage(
                session=_make_session(),
                user_id="user-cost",
                prompt_tokens=200,
                completion_tokens=100,
                cost_usd=0.005,
                model="gpt-4",
                provider="anthropic",
                log_prefix="[SDK]",
            )
            await asyncio.sleep(0)
        mock_log.assert_awaited_once()
        entry = mock_log.call_args[0][0]
        assert entry.user_id == "user-cost"
        assert entry.provider == "anthropic"
        assert entry.model == "gpt-4"
        assert entry.cost_microdollars == 5000
        assert entry.input_tokens == 200
        assert entry.output_tokens == 100
        assert entry.tracking_type == "cost_usd"
        assert entry.metadata["tracking_type"] == "cost_usd"
        assert entry.metadata["tracking_amount"] == 0.005
        assert entry.block_name == "copilot:SDK"
        assert entry.graph_exec_id == "sess-test"

    @pytest.mark.asyncio
    async def test_logs_cost_entry_without_cost_usd(self):
        """When cost_usd is None, tracking_type should be 'tokens'."""
        mock_log = AsyncMock()
        with (
            patch(
                "backend.copilot.token_tracking.record_cost_usage",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.copilot.token_tracking.platform_cost_db",
                return_value=type(
                    "FakePlatformCostDb", (), {"log_platform_cost": mock_log}
                )(),
            ),
        ):
            await persist_and_record_usage(
                session=None,
                user_id="user-tokens",
                prompt_tokens=100,
                completion_tokens=50,
                log_prefix="[Baseline]",
            )
            await asyncio.sleep(0)
        mock_log.assert_awaited_once()
        entry = mock_log.call_args[0][0]
        assert entry.cost_microdollars is None
        assert entry.tracking_type == "tokens"
        assert entry.metadata["tracking_type"] == "tokens"
        assert entry.metadata["tracking_amount"] == 150
        assert entry.graph_exec_id is None
        assert entry.block_name == "copilot:Baseline"

    @pytest.mark.asyncio
    async def test_skips_cost_log_when_no_user_id(self):
        """No PlatformCostLog entry when user_id is None."""
        mock_log = AsyncMock()
        with (
            patch(
                "backend.copilot.token_tracking.record_cost_usage",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.copilot.token_tracking.platform_cost_db",
                return_value=type(
                    "FakePlatformCostDb", (), {"log_platform_cost": mock_log}
                )(),
            ),
        ):
            await persist_and_record_usage(
                session=None,
                user_id=None,
                prompt_tokens=100,
                completion_tokens=50,
            )
            await asyncio.sleep(0)
        mock_log.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cost_usd_invalid_string_falls_back_to_tokens(self):
        """Invalid cost_usd string should fall back to tokens tracking."""
        mock_log = AsyncMock()
        with (
            patch(
                "backend.copilot.token_tracking.record_cost_usage",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.copilot.token_tracking.platform_cost_db",
                return_value=type(
                    "FakePlatformCostDb", (), {"log_platform_cost": mock_log}
                )(),
            ),
        ):
            await persist_and_record_usage(
                session=None,
                user_id="user-invalid",
                prompt_tokens=100,
                completion_tokens=50,
                cost_usd="not-a-number",
            )
            await asyncio.sleep(0)
        mock_log.assert_awaited_once()
        entry = mock_log.call_args[0][0]
        assert entry.cost_microdollars is None
        assert entry.metadata["tracking_type"] == "tokens"

    @pytest.mark.asyncio
    async def test_cost_usd_string_number_is_parsed(self):
        """String-encoded cost_usd (e.g. from OpenRouter) should be parsed."""
        mock_log = AsyncMock()
        with (
            patch(
                "backend.copilot.token_tracking.record_cost_usage",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.copilot.token_tracking.platform_cost_db",
                return_value=type(
                    "FakePlatformCostDb", (), {"log_platform_cost": mock_log}
                )(),
            ),
        ):
            await persist_and_record_usage(
                session=None,
                user_id="user-str",
                prompt_tokens=100,
                completion_tokens=50,
                cost_usd="0.01",
            )
            await asyncio.sleep(0)
        mock_log.assert_awaited_once()
        entry = mock_log.call_args[0][0]
        assert entry.cost_microdollars == 10_000
        assert entry.metadata["tracking_type"] == "cost_usd"

    @pytest.mark.asyncio
    async def test_empty_log_prefix_produces_copilot_block_name(self):
        """Empty log_prefix results in block_name='copilot'."""
        mock_log = AsyncMock()
        with (
            patch(
                "backend.copilot.token_tracking.record_cost_usage",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.copilot.token_tracking.platform_cost_db",
                return_value=type(
                    "FakePlatformCostDb", (), {"log_platform_cost": mock_log}
                )(),
            ),
        ):
            await persist_and_record_usage(
                session=None,
                user_id="user-empty",
                prompt_tokens=10,
                completion_tokens=5,
                log_prefix="",
            )
            await asyncio.sleep(0)
        entry = mock_log.call_args[0][0]
        assert entry.block_name == "copilot"

    @pytest.mark.asyncio
    async def test_cache_tokens_included_in_metadata(self):
        """Cache token counts should be present in the metadata."""
        mock_log = AsyncMock()
        with (
            patch(
                "backend.copilot.token_tracking.record_cost_usage",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.copilot.token_tracking.platform_cost_db",
                return_value=type(
                    "FakePlatformCostDb", (), {"log_platform_cost": mock_log}
                )(),
            ),
        ):
            await persist_and_record_usage(
                session=None,
                user_id="user-cache",
                prompt_tokens=100,
                completion_tokens=50,
                cache_read_tokens=5000,
                cache_creation_tokens=300,
            )
            await asyncio.sleep(0)
        entry = mock_log.call_args[0][0]
        assert entry.metadata["cache_read_tokens"] == 5000
        assert entry.metadata["cache_creation_tokens"] == 300
        assert entry.metadata["source"] == "copilot"

    @pytest.mark.asyncio
    async def test_logs_cost_only_when_tokens_zero(self):
        """Zero prompt+completion tokens with cost_usd set still logs the entry."""
        mock_log = AsyncMock()
        with (
            patch(
                "backend.copilot.token_tracking.record_cost_usage",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.copilot.token_tracking.platform_cost_db",
                return_value=type(
                    "FakePlatformCostDb", (), {"log_platform_cost": mock_log}
                )(),
            ),
        ):
            await persist_and_record_usage(
                session=None,
                user_id="user-cached",
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.005,
                model="claude-3-5-sonnet",
                provider="anthropic",
                log_prefix="[SDK]",
            )
            await asyncio.sleep(0)
        # Guard: total_tokens == 0 but cost_usd is set — must still log
        mock_log.assert_awaited_once()
        entry = mock_log.call_args[0][0]
        assert entry.user_id == "user-cached"
        assert entry.tracking_type == "cost_usd"
        assert entry.cost_microdollars == 5000
        assert entry.input_tokens == 0
        assert entry.output_tokens == 0

    @pytest.mark.asyncio
    async def test_negative_cost_usd_falls_back_to_tokens(self):
        """Negative cost_usd must be rejected — val >= 0 guard in persist_and_record_usage."""
        mock_log = AsyncMock()
        with (
            patch(
                "backend.copilot.token_tracking.record_cost_usage",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.copilot.token_tracking.platform_cost_db",
                return_value=type(
                    "FakePlatformCostDb", (), {"log_platform_cost": mock_log}
                )(),
            ),
        ):
            await persist_and_record_usage(
                session=None,
                user_id="user-negative",
                prompt_tokens=100,
                completion_tokens=50,
                cost_usd=-0.01,
            )
            await asyncio.sleep(0)
        mock_log.assert_awaited_once()
        entry = mock_log.call_args[0][0]
        # Negative cost rejected — falls back to token-based tracking
        assert entry.cost_microdollars is None
        assert entry.metadata["tracking_type"] == "tokens"
