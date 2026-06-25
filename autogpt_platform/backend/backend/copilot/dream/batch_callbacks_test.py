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

    async def fake_eval(script, numkeys, key, token):
        # The only Lua the dream path runs is the lock's single-key
        # compare-and-delete; mirror its semantics on the string store.
        if string_store.get(key) == token:
            string_store.pop(key, None)
            return 1
        return 0

    stub = AsyncMock()
    stub.hset.side_effect = fake_hset
    stub.hgetall.side_effect = fake_hgetall
    stub.get.side_effect = fake_get
    stub.set.side_effect = fake_set
    stub.expire.side_effect = fake_expire
    stub.delete.side_effect = fake_delete
    stub.eval.side_effect = fake_eval

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
            "phase_models": {
                "consolidate": "claude-sonnet-4-6",
                "recombine": "claude-opus-4-7",
                "sanitize": "claude-sonnet-4-6",
            },
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

        _, _, string_store = fake_redis
        # The orchestrator holds the dream lock while persisting the bundle;
        # the bundle captures the lock's ownership token for the callback.
        string_store["dream:inflight:u1"] = "tok-u1"
        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1",
                group_id="user_u1",
                window_start=now,
                window_end=now,
                known_fact_uuids={"fact-1"},
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
        release_lock = AsyncMock()
        with patch("backend.copilot.dream.apply.apply_operations", apply), patch(
            "backend.copilot.dream.job_status.mark_complete", mark_complete
        ), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", record_cost
        ), patch(
            "backend.copilot.dream.batch_callbacks.release_dream_lock", release_lock
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        apply.assert_awaited_once()
        # The demotion allowlist is threaded from the bundle already loaded
        # for the clamp — apply must not re-read the bundle from Redis.
        assert apply.call_args.kwargs["known_fact_uuids"] == {"fact-1"}
        mark_complete.assert_awaited_once()
        # The sanitizer's user-facing narrative must ride on the result so the
        # Memory Visualizer isn't blank for batch-completed dreams.
        assert mark_complete.call_args.kwargs["result"].summary_for_user == "ok"
        # The batch path disowned the dream lock to this callback; the
        # terminal handler must release it with the ownership token the
        # input bundle carried — compare-and-delete, never a blind DEL.
        release_lock.assert_awaited_once_with("u1", "tok-u1")
        # One cost-log row per phase (consolidate, recombine, sanitize)
        assert record_cost.await_count == 3
        for call in record_cost.await_args_list:
            assert call.kwargs["execution_path"] == "anthropic_batch"
        # Each phase is priced with ITS OWN model — recombine uses the
        # advanced (opus) model, not phase 1's standard model.
        models_by_phase = {
            call.kwargs["phase_usage"].phase: call.kwargs["phase_usage"].model
            for call in record_cost.await_args_list
        }
        assert models_by_phase["consolidate"] == "claude-sonnet-4-6"
        assert models_by_phase["recombine"] == "claude-opus-4-7"
        assert models_by_phase["sanitize"] == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_redispatch_after_charge_does_not_double_bill(self, fake_redis):
        """If the BatchExecutor crashes between charging and
        ``remove_pending``, the next walk re-dispatches the same batch.
        The Redis dedup gates must prevent BOTH the second charge and a
        second ``apply_operations`` run."""
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
        # And the memory writes ran exactly once — the apply gate ate the
        # duplicate delivery.
        apply.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_duplicate_batch_dispatch_skips_apply(self, fake_redis):
        """Executor crash between ``apply_operations`` returning and the
        state cleanup re-dispatches the sanitize batch with all per-pass
        state intact. The ``dream:applied:{pass_id}`` SETNX gate must keep
        apply at-most-once — otherwise every consolidated fact and proposal
        is written to the user's graph a second time as fresh episodes —
        while the duplicate skips mark_complete (preserving the first
        delivery's job result) and still releases the lock + cleans up
        state."""
        from backend.copilot.dream.batch_callbacks import _write_phase_to_state
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        _, _, string_store = fake_redis
        string_store["dream:inflight:u1"] = "tok-u1"
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
        release_lock = AsyncMock()
        # Crash-before-cleanup simulation: state + input bundle survive the
        # first delivery, so the re-dispatch sees a fully populated pass.
        with patch("backend.copilot.dream.apply.apply_operations", apply), patch(
            "backend.copilot.dream.job_status.mark_complete", mark_complete
        ), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", AsyncMock()
        ), patch(
            "backend.copilot.dream.batch_callbacks.release_dream_lock", release_lock
        ), patch(
            "backend.copilot.dream.batch_callbacks._delete_state", AsyncMock()
        ), patch(
            "backend.copilot.dream.batch_callbacks.delete_input_bundle", AsyncMock()
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )
            # Re-dispatch of the same finished batch.
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        apply.assert_awaited_once()
        # The gate key is what makes the dedup stick across processes.
        assert "dream:applied:p1" in string_store
        # Only the first delivery writes the job result — the duplicate
        # must NOT call mark_complete with empty apply_stats, or it would
        # zero the real consolidated/proposal/demotion counts and
        # dream_session_id the first delivery recorded.
        mark_complete.assert_awaited_once()
        assert mark_complete.call_args.kwargs["result"].summary_for_user == "ok"
        # Both deliveries release the lock (token CAS makes the second a
        # safe no-op) and clean up state.
        assert release_lock.await_count == 2
        for call in release_lock.await_args_list:
            assert call.args == ("u1", "tok-u1")

    @pytest.mark.asyncio
    async def test_apply_gate_redis_outage_fails_pass_not_silent_success(
        self, fake_redis
    ):
        """A Redis outage while claiming the apply gate means we cannot
        distinguish first delivery from duplicate. The pass must be marked
        errored — completing it would report success while no memory was
        written, silently dropping the dream."""
        from backend.copilot.dream.batch_callbacks import _write_phase_to_state

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

        apply = AsyncMock()
        mark_complete = AsyncMock()
        mark_errored = AsyncMock()
        gate = AsyncMock(return_value="error")
        with patch("backend.copilot.dream.apply.apply_operations", apply), patch(
            "backend.copilot.dream.job_status.mark_complete", mark_complete
        ), patch("backend.copilot.dream.job_status.mark_errored", mark_errored), patch(
            "backend.copilot.dream.batch_callbacks._claim_apply_gate", gate
        ), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", AsyncMock()
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        apply.assert_not_awaited()
        mark_complete.assert_not_awaited()
        mark_errored.assert_awaited_once()
        assert "gate unavailable" in mark_errored.call_args.kwargs["error"]


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_malformed_payload_marks_job_errored_not_stuck(self, fake_redis):
        """A payload missing pass_id/phase is a dead end the normal fail
        path can't reach — the admin row must still go terminal instead of
        sitting queued/submitted until its TTL."""
        from backend.copilot.dream.job_status import read_status, write_initial_status

        await write_initial_status(kind="dream_pass", job_id="j-dead", user_id="u1")
        entry = _entry(phase="consolidate")
        entry.payload = {"user_id": "u1", "job_id": "j-dead"}

        with patch(
            "backend.copilot.dream.batch_callbacks.release_dream_lock", AsyncMock()
        ):
            await handle_dream_batch_result(entry, [])

        final = await read_status(kind="dream_pass", job_id="j-dead")
        assert final is not None
        assert final.state == "errored"
        assert "missing" in (final.error or "")

    @pytest.mark.asyncio
    async def test_unknown_phase_marks_job_errored_not_stuck(self, fake_redis):
        from backend.copilot.dream.job_status import read_status, write_initial_status

        await write_initial_status(kind="dream_pass", job_id="j-odd", user_id="u1")
        entry = _entry(phase="consolidate")
        entry.payload = {
            "user_id": "u1",
            "pass_id": "p1",
            "job_id": "j-odd",
            "phase": "daydream",
        }

        with patch(
            "backend.copilot.dream.batch_callbacks.release_dream_lock", AsyncMock()
        ):
            await handle_dream_batch_result(entry, [])

        final = await read_status(kind="dream_pass", job_id="j-odd")
        assert final is not None
        assert final.state == "errored"
        assert "unknown batch phase" in (final.error or "")

    @pytest.mark.asyncio
    async def test_errored_row_short_circuits_to_mark_errored(self, fake_redis):
        mark_errored = AsyncMock()
        record_cost = AsyncMock()
        with patch(
            "backend.copilot.dream.job_status.mark_errored", mark_errored
        ), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", record_cost
        ):
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
        # The errored phase itself is not billed.
        record_cost.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_json_marks_errored_does_not_chain(self, fake_redis):
        """A row whose content is not parseable JSON must NOT chain to
        the next phase — the next phase's prompt would be built on
        garbage. Pydantic's default ``model_validate`` is permissive
        about unknown fields (extra="ignore"), so the failure mode the
        test pins is invalid-JSON not unknown-field."""
        submit_phase = AsyncMock()
        mark_errored = AsyncMock()
        record_cost = AsyncMock()
        with patch(
            "backend.copilot.dream.batch_callbacks.submit_phase", submit_phase
        ), patch("backend.copilot.dream.job_status.mark_errored", mark_errored), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", record_cost
        ):
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
    async def test_apply_crash_marks_errored_still_records_usage(self, fake_redis):
        """If apply raises, the pass is errored — but the three LLM phases
        already ran and Anthropic billed us, so their usage is still
        recorded (matches the sync path + dream/billing.py)."""
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
        # consolidate + recombine + sanitize all completed and were billed
        # by Anthropic; apply failing afterward doesn't refund those tokens.
        assert record_cost.await_count == 3

    @pytest.mark.asyncio
    async def test_unexpected_crash_releases_disowned_lock(self, fake_redis):
        """An unexpected raise OUTSIDE the handler's own _fail_pass guards
        (here phase-chaining blows up) must still release the disowned dream
        lock and mark the job errored. BatchExecutor._dispatch swallows
        handler exceptions, so without the crash guard this would strand the
        user behind the lock until its extended TTL."""
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        _, _, string_store = fake_redis
        string_store["dream:inflight:u1"] = "tok-u1"
        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1", group_id="user_u1", window_start=now, window_end=now
            ),
        )

        chain = AsyncMock(side_effect=RuntimeError("redis exploded mid-chain"))
        mark_errored = AsyncMock()
        record_cost = AsyncMock()
        release_lock = AsyncMock()
        with patch(
            "backend.copilot.dream.batch_callbacks._chain_next_phase", chain
        ), patch("backend.copilot.dream.job_status.mark_errored", mark_errored), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", record_cost
        ), patch(
            "backend.copilot.dream.batch_callbacks.release_dream_lock", release_lock
        ):
            # Must not propagate — the crash guard finalizes and swallows.
            await handle_dream_batch_result(
                _entry(phase="consolidate"),
                [_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT)],
            )

        mark_errored.assert_awaited_once()
        assert "handler crashed" in mark_errored.call_args.kwargs["error"]
        # Released with the ownership token the input bundle carried.
        release_lock.assert_awaited_once_with("u1", "tok-u1")


class TestMalformedPayload:
    @pytest.mark.asyncio
    async def test_missing_pass_id_releases_lock_without_token(self, fake_redis):
        """Malformed payload (missing pass_id) early-returns but, since the
        user_id is known, still attempts the lock release — with no pass_id
        there's no persisted token to read, so the release is token-less
        (release_dream_lock then defers to the lock TTL rather than
        blind-deleting)."""
        release = AsyncMock()
        entry = _entry()
        entry.payload["pass_id"] = ""
        with patch("backend.copilot.dream.batch_callbacks.release_dream_lock", release):
            await handle_dream_batch_result(
                entry,
                [_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT)],
            )
        release.assert_awaited_once_with("u1", None)

    @pytest.mark.asyncio
    async def test_unknown_phase_label_releases_lock_with_persisted_token(
        self, fake_redis
    ):
        """An unknown phase early-returns but must still release the dream
        lock the orchestrator disowned to this callback — using the token
        the input bundle carries for this pass."""
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        _, _, string_store = fake_redis
        string_store["dream:inflight:u1"] = "tok-u1"
        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1", group_id="user_u1", window_start=now, window_end=now
            ),
        )

        release = AsyncMock()
        entry = _entry()
        entry.payload["phase"] = "some_fake_phase"
        with patch("backend.copilot.dream.batch_callbacks.release_dream_lock", release):
            await handle_dream_batch_result(entry, [_row(custom_id="x", content="y")])
        release.assert_awaited_once_with("u1", "tok-u1")


class TestLockTokenWiring:
    """End-to-end token flow with the real ``release_dream_lock`` — the
    fake redis implements the single-key compare-and-delete Lua."""

    async def _seed_terminal_pass(self) -> None:
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

    async def _dispatch_sanitize(self) -> None:
        with patch(
            "backend.copilot.dream.apply.apply_operations",
            AsyncMock(return_value={"writes": 0}),
        ), patch("backend.copilot.dream.job_status.mark_complete", AsyncMock()), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", AsyncMock()
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

    @pytest.mark.asyncio
    async def test_terminal_release_deletes_lock_when_token_matches(self, fake_redis):
        _, _, string_store = fake_redis
        string_store["dream:inflight:u1"] = "tok-u1"
        await self._seed_terminal_pass()

        await self._dispatch_sanitize()

        assert "dream:inflight:u1" not in string_store

    @pytest.mark.asyncio
    async def test_token_read_failure_after_complete_keeps_job_completed(
        self, fake_redis
    ):
        """A Redis blip on the lock-token read in the terminal tail fires
        AFTER mark_complete already ran. The read must stay best-effort
        (token=None → lock TTL fallback) — letting it propagate would hit
        the handler's crash guard, whose _fail_pass rewrites the
        already-completed job to errored."""
        _, _, string_store = fake_redis
        string_store["dream:inflight:u1"] = "tok-u1"
        await self._seed_terminal_pass()

        mark_complete = AsyncMock()
        mark_errored = AsyncMock()
        release_lock = AsyncMock()
        read_token = AsyncMock(side_effect=ConnectionError("redis blip"))
        with patch(
            "backend.copilot.dream.apply.apply_operations",
            AsyncMock(return_value={"writes": 0}),
        ), patch(
            "backend.copilot.dream.job_status.mark_complete", mark_complete
        ), patch(
            "backend.copilot.dream.job_status.mark_errored", mark_errored
        ), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", AsyncMock()
        ), patch(
            "backend.copilot.dream.batch_callbacks.read_lock_token", read_token
        ), patch(
            "backend.copilot.dream.batch_callbacks.release_dream_lock", release_lock
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        mark_complete.assert_awaited_once()
        mark_errored.assert_not_awaited()
        # Token read failed → token-less release; release_dream_lock then
        # defers to the lock TTL rather than blind-deleting.
        release_lock.assert_awaited_once_with("u1", None)

    @pytest.mark.asyncio
    async def test_cleanup_failure_after_complete_keeps_job_completed(self, fake_redis):
        """A Redis blip on the post-mark_complete state/bundle deletes must
        not route through the crash guard to _fail_pass — both keys carry
        24h TTLs, so cleanup is best-effort and the completed job stays
        completed."""
        await self._seed_terminal_pass()

        mark_complete = AsyncMock()
        mark_errored = AsyncMock()
        delete_state = AsyncMock(side_effect=ConnectionError("redis blip"))
        with patch(
            "backend.copilot.dream.apply.apply_operations",
            AsyncMock(return_value={"writes": 0}),
        ), patch(
            "backend.copilot.dream.job_status.mark_complete", mark_complete
        ), patch(
            "backend.copilot.dream.job_status.mark_errored", mark_errored
        ), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", AsyncMock()
        ), patch(
            "backend.copilot.dream.batch_callbacks._delete_state", delete_state
        ), patch(
            "backend.copilot.dream.batch_callbacks.release_dream_lock", AsyncMock()
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        delete_state.assert_awaited_once()
        mark_complete.assert_awaited_once()
        mark_errored.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_duplicate_dispatch_finalizes_job_stuck_before_mark_complete(
        self, fake_redis
    ):
        """First delivery crashed between apply and mark_complete: the gate
        is claimed but the job row never went terminal. The duplicate must
        finalize the row (with the clamped op counts) instead of leaving it
        stuck in 'submitted' until the row TTL."""
        from backend.copilot.dream.job_status import (
            read_status,
            update_status_phase,
            write_initial_status,
        )

        _, _, string_store = fake_redis
        await self._seed_terminal_pass()
        # Gate already claimed by the crashed first delivery.
        string_store["dream:applied:p1"] = "1"
        await write_initial_status(kind="dream_pass", job_id="j1", user_id="u1")
        await update_status_phase(kind="dream_pass", job_id="j1", state="submitted")

        apply = AsyncMock()
        with patch("backend.copilot.dream.apply.apply_operations", apply), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", AsyncMock()
        ), patch(
            "backend.copilot.dream.batch_callbacks.release_dream_lock", AsyncMock()
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        apply.assert_not_awaited()
        final = await read_status(kind="dream_pass", job_id="j1")
        assert final is not None
        assert final.state == "complete"
        assert final.result is not None
        # Counts are attempted ops, not confirmed apply results — the
        # summary must say so.
        assert final.result["summary_for_user"].endswith("ok")
        assert "duplicate delivery" in final.result["summary_for_user"]

    @pytest.mark.asyncio
    async def test_duplicate_dispatch_leaves_terminal_job_untouched(self, fake_redis):
        """A duplicate against an already-completed row must not rewrite it
        — the first delivery's real apply stats stay authoritative."""
        _, _, string_store = fake_redis
        await self._seed_terminal_pass()
        string_store["dream:applied:p1"] = "1"

        mark_complete = AsyncMock()
        read_existing = AsyncMock(return_value=MagicMock(state="complete"))
        with patch("backend.copilot.dream.apply.apply_operations", AsyncMock()), patch(
            "backend.copilot.dream.job_status.mark_complete", mark_complete
        ), patch("backend.copilot.dream.job_status.read_status", read_existing), patch(
            "backend.copilot.dream.batch_callbacks.record_phase_cost", AsyncMock()
        ), patch(
            "backend.copilot.dream.batch_callbacks.release_dream_lock", AsyncMock()
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        mark_complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_terminal_release_leaves_lock_reacquired_by_newer_pass(
        self, fake_redis
    ):
        """The blocker scenario: this pass's lock expired mid-batch and a
        NEWER pass re-acquired the key with its own token. The late callback
        must not delete the new holder's lock — that would let a third
        concurrent pass start."""
        _, _, string_store = fake_redis
        string_store["dream:inflight:u1"] = "tok-u1"
        await self._seed_terminal_pass()
        # Simulate expiry + re-acquire by a newer pass between submit and
        # the (late) terminal callback.
        string_store["dream:inflight:u1"] = "tok-newer-pass"

        await self._dispatch_sanitize()

        assert string_store["dream:inflight:u1"] == "tok-newer-pass"
