"""Tests for the dream-pass batch result handler (sequential chain).

Covers the namespace handler that the BatchExecutor invokes when a
dream batch lands. Dream phases are sequentially dependent so the
handler chains: phase 1 result → submit phase 2 → phase 2 result →
submit phase 3 → phase 3 result → apply + mark JobStatus complete.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.dream.batch_callbacks import handle_dream_batch_result
from backend.executor.batch_executor import PendingEntry
from backend.util.llm.providers import BatchResultRow


@pytest.fixture
def fake_redis():
    """In-memory redis fixture matching the BatchExecutor / JobStatus pattern."""
    store: dict[str, dict[str, str]] = {}
    string_store: dict[str, str] = {}

    async def fake_hset(key, field, value):
        store.setdefault(key, {})[field] = value
        return 1

    async def fake_hgetall(key):
        return store.get(key, {})

    async def fake_get(key):
        return string_store.get(key)

    async def fake_set(key, value, ex=None, nx=False):
        # Mirror redis.set semantics: when ``nx=True`` and the key
        # already exists, set returns None and leaves the existing
        # value alone. This is what makes the dedup gate idempotent.
        if nx and key in string_store:
            return None
        string_store[key] = value
        return True

    async def fake_delete(key):
        store.pop(key, None)
        string_store.pop(key, None)

    async def fake_expire(key, ttl):
        return 1

    stub = AsyncMock()
    stub.hset.side_effect = fake_hset
    stub.hgetall.side_effect = fake_hgetall
    stub.get.side_effect = fake_get
    stub.set.side_effect = fake_set
    stub.expire.side_effect = fake_expire
    stub.delete.side_effect = fake_delete

    async def fake_get_redis_async():
        return stub

    with patch(
        "backend.data.redis_client.get_redis_async",
        side_effect=fake_get_redis_async,
    ):
        yield stub, store, string_store


def _entry(
    *, pass_id: str = "p1", job_id: str = "j1", phase: str = "consolidate"
) -> PendingEntry:
    now = datetime.now(timezone.utc)
    custom_id = f"{pass_id}:{phase}"
    return PendingEntry(
        provider="anthropic",
        provider_batch_id=f"msgbatch_{phase}",
        callback_namespace="dream_pass",
        submitted_at=now,
        next_poll_at=now,
        payload={
            "user_id": "u1",
            "pass_id": pass_id,
            "job_id": job_id,
            "phase": phase,
            "model": "claude-sonnet-4-6",
            "custom_ids": [custom_id],
            "phase_for_custom_id": {custom_id: phase},
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


# Valid Pydantic content per phase
_CONSOLIDATE_CONTENT = '{"facts": []}'
_RECOMBINE_CONTENT = '{"proposals": []}'
_SANITIZE_CONTENT = (
    '{"writes": [], "proposals": [], "demotions": [], '
    '"entity_invalidations": [], "summary_for_user": "ok"}'
)


class TestPhaseChaining:
    @pytest.mark.asyncio
    async def test_consolidate_result_submits_recombine_batch(self, fake_redis):
        """Phase 1 result must trigger phase 2 submission, NOT apply."""
        # Persist a fake input bundle the chain can read back.
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1",
                group_id="user_u1",
                window_start=now,
                window_end=now,
            ),
        )

        submit_phase = AsyncMock(
            return_value=MagicMock(provider_batch_id="msgbatch_recombine")
        )
        update_status = AsyncMock()

        with patch(
            "backend.copilot.dream.batch_callbacks.submit_phase", submit_phase
        ), patch(
            "backend.copilot.dream.job_status.update_status_phase", update_status
        ), patch(
            "backend.copilot.dream.batch_callbacks._anthropic_api_key",
            return_value="sk-ant-test",
        ):
            await handle_dream_batch_result(
                _entry(phase="consolidate"),
                [_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT)],
            )

        submit_phase.assert_awaited_once()
        kwargs = submit_phase.call_args.kwargs
        assert kwargs["phase"] == "recombine"
        # Phase 2 prompt builder needs phase 1's output
        assert kwargs["consolidated_json"] == _CONSOLIDATE_CONTENT
        # JobStatus advanced
        update_status.assert_awaited_once()
        update_kwargs = update_status.call_args.kwargs
        assert update_kwargs["current_phase"] == "recombine"

    @pytest.mark.asyncio
    async def test_recombine_result_submits_sanitize_batch(self, fake_redis):
        """Phase 2 result chains to phase 3 with BOTH prior phases' outputs."""
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1", group_id="user_u1", window_start=now, window_end=now
            ),
        )

        # Pre-seed phase 1's output in the per-pass state
        from backend.copilot.dream.batch_callbacks import _write_phase_to_state

        await _write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )

        submit_phase = AsyncMock(
            return_value=MagicMock(provider_batch_id="msgbatch_sanitize")
        )
        with patch(
            "backend.copilot.dream.batch_callbacks.submit_phase", submit_phase
        ), patch(
            "backend.copilot.dream.batch_callbacks._anthropic_api_key",
            return_value="sk-ant-test",
        ), patch(
            "backend.copilot.dream.job_status.update_status_phase", AsyncMock()
        ):
            await handle_dream_batch_result(
                _entry(phase="recombine"),
                [_row(custom_id="p1:recombine", content=_RECOMBINE_CONTENT)],
            )

        kwargs = submit_phase.call_args.kwargs
        assert kwargs["phase"] == "sanitize"
        assert kwargs["consolidated_json"] == _CONSOLIDATE_CONTENT
        assert kwargs["recombined_json"] == _RECOMBINE_CONTENT

    @pytest.mark.asyncio
    async def test_sanitize_result_runs_apply_marks_complete_logs_costs(
        self, fake_redis
    ):
        """Phase 3 is terminal: apply runs, all three phases logged at
        anthropic_batch path (50% discount in the rate card), JobStatus
        flips to complete."""
        from backend.copilot.dream.batch_callbacks import _write_phase_to_state
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1", group_id="user_u1", window_start=now, window_end=now
            ),
        )
        await _write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        await _write_phase_to_state(
            pass_id="p1",
            phase="recombine",
            row=_row(custom_id="p1:recombine", content=_RECOMBINE_CONTENT),
        )

        apply = AsyncMock(return_value={"writes": 0, "snapshot": "..."})
        mark_complete = AsyncMock()
        record_cost = AsyncMock()
        with patch("backend.copilot.dream.apply.apply_operations", apply), patch(
            "backend.copilot.dream.job_status.mark_complete", mark_complete
        ), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", record_cost
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        apply.assert_awaited_once()
        mark_complete.assert_awaited_once()
        # One cost-log row per phase (consolidate, recombine, sanitize)
        assert record_cost.await_count == 3
        for call in record_cost.await_args_list:
            assert call.kwargs["execution_path"] == "anthropic_batch"

    @pytest.mark.asyncio
    async def test_redispatch_after_charge_does_not_double_bill(self, fake_redis):
        """If the BatchExecutor crashes between charging and
        ``remove_pending``, the next walk re-dispatches the same batch.
        The Redis dedup gate must prevent the second charge while
        still letting apply re-run idempotently."""
        from backend.copilot.dream.batch_callbacks import _write_phase_to_state
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1", group_id="user_u1", window_start=now, window_end=now
            ),
        )
        await _write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        await _write_phase_to_state(
            pass_id="p1",
            phase="recombine",
            row=_row(custom_id="p1:recombine", content=_RECOMBINE_CONTENT),
        )

        apply = AsyncMock(return_value={"writes": 0, "snapshot": "..."})
        record_cost = AsyncMock()
        with patch("backend.copilot.dream.apply.apply_operations", apply), patch(
            "backend.copilot.dream.job_status.mark_complete", AsyncMock()
        ), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", record_cost
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )
            # Simulated re-dispatch: BatchExecutor crashed between
            # charge + remove_pending, walks the queue again, calls us
            # a second time with the same batch result.
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        # 3 phases charged on the first call, ZERO on the re-dispatch.
        assert record_cost.await_count == 3


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_errored_row_short_circuits_to_mark_errored(self, fake_redis):
        mark_errored = AsyncMock()
        with patch("backend.copilot.dream.job_status.mark_errored", mark_errored):
            await handle_dream_batch_result(
                _entry(phase="consolidate"),
                [
                    _row(
                        custom_id="p1:consolidate",
                        content="",
                        error="content moderation",
                    )
                ],
            )
        mark_errored.assert_awaited_once()
        assert "content moderation" in mark_errored.call_args.kwargs["error"]

    @pytest.mark.asyncio
    async def test_invalid_json_marks_errored_does_not_chain(self, fake_redis):
        """A row whose content is not parseable JSON must NOT chain to
        the next phase — the next phase's prompt would be built on
        garbage. Pydantic's default ``model_validate`` is permissive
        about unknown fields (extra="ignore"), so the failure mode the
        test pins is invalid-JSON not unknown-field."""
        submit_phase = AsyncMock()
        mark_errored = AsyncMock()
        with patch(
            "backend.copilot.dream.batch_callbacks.submit_phase", submit_phase
        ), patch("backend.copilot.dream.job_status.mark_errored", mark_errored):
            await handle_dream_batch_result(
                _entry(phase="consolidate"),
                [
                    _row(
                        custom_id="p1:consolidate",
                        content="this is not JSON at all",
                    )
                ],
            )
        submit_phase.assert_not_awaited()
        mark_errored.assert_awaited_once()
        assert "invalid output shape" in mark_errored.call_args.kwargs["error"]

    @pytest.mark.asyncio
    async def test_apply_crash_marks_errored_skips_cost_log(self, fake_redis):
        """If apply raises, the user must NOT be charged — those tokens
        bought operations that didn't land in the graph."""
        from backend.copilot.dream.batch_callbacks import _write_phase_to_state
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1", group_id="user_u1", window_start=now, window_end=now
            ),
        )
        await _write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        await _write_phase_to_state(
            pass_id="p1",
            phase="recombine",
            row=_row(custom_id="p1:recombine", content=_RECOMBINE_CONTENT),
        )

        apply = AsyncMock(side_effect=RuntimeError("FalkorDB unreachable"))
        mark_errored = AsyncMock()
        record_cost = AsyncMock()
        with patch("backend.copilot.dream.apply.apply_operations", apply), patch(
            "backend.copilot.dream.job_status.mark_errored", mark_errored
        ), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", record_cost
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        mark_errored.assert_awaited_once()
        record_cost.assert_not_awaited()


class TestMalformedPayload:
    @pytest.mark.asyncio
    async def test_missing_pass_id_drops_silently(self, fake_redis):
        entry = _entry()
        entry.payload["pass_id"] = ""
        # Should NOT crash even though payload is malformed
        await handle_dream_batch_result(
            entry,
            [_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT)],
        )

    @pytest.mark.asyncio
    async def test_unknown_phase_label_drops_silently(self, fake_redis):
        entry = _entry()
        entry.payload["phase"] = "some_fake_phase"
        await handle_dream_batch_result(entry, [_row(custom_id="x", content="y")])
