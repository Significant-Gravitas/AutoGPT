"""Tests for the dream-pass batch result handler.

Covers the namespace handler that the BatchExecutor invokes when a
dream batch lands:
  * Per-phase accumulator round-trip
  * Single-phase result doesn't trigger apply (still waiting on the rest)
  * All-three-phases result calls apply + marks JobStatus complete
  * Errored phase short-circuits to mark JobStatus errored
  * Apply crash marks JobStatus errored
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.dream.batch_callbacks import handle_dream_batch_result
from backend.executor.batch_executor import PendingEntry
from backend.util.llm.providers import BatchResultRow


@pytest.fixture
def fake_redis():
    """Same in-memory redis fixture pattern as job_status_test.py."""
    store: dict[str, dict[str, str]] = {}

    async def fake_hset(key, field, value):
        store.setdefault(key, {})[field] = value
        return 1

    async def fake_hgetall(key):
        return store.get(key, {})

    async def fake_expire(key, ttl):
        return 1

    async def fake_delete(key):
        store.pop(key, None)

    stub = AsyncMock()
    stub.hset.side_effect = fake_hset
    stub.hgetall.side_effect = fake_hgetall
    stub.expire.side_effect = fake_expire
    stub.delete.side_effect = fake_delete

    async def fake_get_redis_async():
        return stub

    with patch(
        "backend.data.redis_client.get_redis_async",
        side_effect=fake_get_redis_async,
    ):
        yield stub, store


def _entry(*, pass_id: str = "p1", job_id: str = "j1") -> PendingEntry:
    now = datetime.now(timezone.utc)
    return PendingEntry(
        provider="anthropic",
        provider_batch_id="msgbatch_1",
        callback_namespace="dream_pass",
        submitted_at=now,
        next_poll_at=now,
        payload={
            "user_id": "u1",
            "pass_id": pass_id,
            "job_id": job_id,
            "phase_for_custom_id": {
                f"{pass_id}:consolidate": "consolidate",
                f"{pass_id}:recombine": "recombine",
                f"{pass_id}:sanitize": "sanitize",
            },
        },
    )


def _row(*, custom_id: str, content: str, error: str | None = None) -> BatchResultRow:
    return BatchResultRow(
        custom_id=custom_id,
        content=content,
        input_tokens=10,
        output_tokens=20,
        error=error,
    )


_SANITIZED_OPS_JSON = (
    '{"writes": [], "proposals": [], "demotions": [], '
    '"entity_invalidations": [], "summary_for_user": ""}'
)


class TestPartialResult:
    @pytest.mark.asyncio
    async def test_single_phase_does_not_call_apply_or_mark_complete(self, fake_redis):
        """One phase landing means two more are still in flight; we
        must NOT call apply or mark the job complete yet."""
        mark_complete = AsyncMock()
        apply = AsyncMock()
        with patch(
            "backend.copilot.dream.job_status.mark_complete",
            mark_complete,
        ), patch(
            "backend.copilot.dream.apply.apply_operations",
            apply,
        ):
            await handle_dream_batch_result(
                _entry(),
                [_row(custom_id="p1:consolidate", content='{"facts": []}')],
            )
        mark_complete.assert_not_awaited()
        apply.assert_not_awaited()


class TestErroredPhase:
    @pytest.mark.asyncio
    async def test_errored_row_short_circuits_to_mark_errored(self, fake_redis):
        """Any phase erroring forces the whole pass into a failed
        terminal state — no point spending compute on the other phases
        when the result will be thrown away."""
        mark_errored = AsyncMock()
        with patch("backend.copilot.dream.job_status.mark_errored", mark_errored):
            await handle_dream_batch_result(
                _entry(),
                [
                    _row(
                        custom_id="p1:consolidate",
                        content="",
                        error="content moderation",
                    )
                ],
            )
        mark_errored.assert_awaited_once()
        kwargs = mark_errored.call_args.kwargs
        assert kwargs["kind"] == "dream_pass"
        assert kwargs["job_id"] == "j1"
        assert "content moderation" in kwargs["error"]


class TestCompletePass:
    @pytest.mark.asyncio
    async def test_three_phases_call_apply_and_mark_complete(self, fake_redis):
        """Once all three phases have landed, the handler:
        1. Validates sanitize output as DreamOperations
        2. Calls apply_operations
        3. Logs phase costs
        4. Marks JobStatus complete
        """
        apply = AsyncMock(return_value={"writes": 0, "snapshot": "..."})
        mark_complete = AsyncMock()
        record_cost = AsyncMock()
        # Three handler invocations (one per phase)
        for cid, phase, content in [
            ("p1:consolidate", "consolidate", '{"facts": []}'),
            ("p1:recombine", "recombine", '{"proposals": []}'),
        ]:
            await handle_dream_batch_result(
                _entry(), [_row(custom_id=cid, content=content)]
            )

        # All three are present after the third dispatch — patch apply
        # only for the terminal call.
        with patch("backend.copilot.dream.apply.apply_operations", apply), patch(
            "backend.copilot.dream.job_status.mark_complete", mark_complete
        ), patch("backend.copilot.dream.billing.record_phase_cost", record_cost):
            await handle_dream_batch_result(
                _entry(),
                [_row(custom_id="p1:sanitize", content=_SANITIZED_OPS_JSON)],
            )

        apply.assert_awaited_once()
        mark_complete.assert_awaited_once()
        # One cost-log row per phase
        assert record_cost.await_count == 3
        # All three logged as anthropic_batch path (50% discount applies)
        for call in record_cost.await_args_list:
            assert call.kwargs["execution_path"] == "anthropic_batch"

    @pytest.mark.asyncio
    async def test_apply_crash_marks_errored_and_skips_cost_log(self, fake_redis):
        """If apply raises mid-write, we MUST NOT charge the user —
        the 50%-discounted cost is moot when the operations didn't
        land in the graph."""
        apply = AsyncMock(side_effect=RuntimeError("FalkorDB unreachable"))
        mark_errored = AsyncMock()
        record_cost = AsyncMock()

        for cid, phase, content in [
            ("p1:consolidate", "consolidate", '{"facts": []}'),
            ("p1:recombine", "recombine", '{"proposals": []}'),
        ]:
            await handle_dream_batch_result(
                _entry(), [_row(custom_id=cid, content=content)]
            )

        with patch("backend.copilot.dream.apply.apply_operations", apply), patch(
            "backend.copilot.dream.job_status.mark_errored", mark_errored
        ), patch("backend.copilot.dream.billing.record_phase_cost", record_cost):
            await handle_dream_batch_result(
                _entry(),
                [_row(custom_id="p1:sanitize", content=_SANITIZED_OPS_JSON)],
            )

        mark_errored.assert_awaited_once()
        record_cost.assert_not_awaited()


class TestMalformedPayload:
    @pytest.mark.asyncio
    async def test_missing_user_id_drops_silently(self, fake_redis):
        """The handler must NOT crash if some upstream wrote a
        malformed payload — log + bail so the rest of the BatchExecutor
        walk continues."""
        entry = _entry()
        entry.payload["user_id"] = ""  # blank
        await handle_dream_batch_result(
            entry, [_row(custom_id="p1:consolidate", content='{"facts": []}')]
        )
        # Just confirm no exception escaped.

    @pytest.mark.asyncio
    async def test_unknown_custom_id_skipped(self, fake_redis):
        """A row whose ``custom_id`` doesn't appear in the
        ``phase_for_custom_id`` mapping is logged + dropped, not
        re-routed to a random phase."""
        mark_errored = AsyncMock()
        with patch("backend.copilot.dream.job_status.mark_errored", mark_errored):
            await handle_dream_batch_result(
                _entry(),
                [
                    _row(
                        custom_id="completely-different-id",
                        content="{}",
                    )
                ],
            )
        mark_errored.assert_not_awaited()
