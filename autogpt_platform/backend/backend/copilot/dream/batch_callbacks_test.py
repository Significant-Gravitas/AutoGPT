"""Tests for the dream-pass batch result handler (sequential chain).

Covers the namespace handler that the BatchExecutor invokes when a
dream batch lands. Dream phases are sequentially dependent so the
handler chains: phase 1 result → submit phase 2 → phase 2 result →
submit phase 3 → phase 3 result → apply + mark JobStatus complete.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)

from backend.copilot.dream import job_status
from backend.copilot.dream.batch_callbacks import handle_dream_batch_result
from backend.copilot.dream.batch_state import (
    read_state,
    state_key,
    write_phase_to_state,
)
from backend.copilot.dream.batch_submit import input_bundle_key, persist_input_bundle
from backend.copilot.dream.cancel import cancel_dream_pass
from backend.copilot.dream.fetch import DreamInput
from backend.copilot.dream.job_status import read_status, write_initial_status
from backend.copilot.dream.pass_record import dream_pass_result_from_row, expired
from backend.copilot.dream.schemas import (
    DreamOperations,
    DreamOperationsSnapshot,
    DreamPhase,
    IngestionDrainStatus,
)
from backend.copilot.dream.store import write_stop
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.trial_cost_context import get_trial_cost_context
from backend.copilot.usage_activation import UsageActivationUnavailable
from backend.data.dream_pass_models import DreamPassDraft
from backend.executor.batch_executor import (
    PendingEntry,
    _dispatch,
    enqueue_pending,
    register_handler,
    walk_once,
)
from backend.util.llm.providers import BatchResultRow


@pytest.fixture(autouse=True)
def submitted_rows(fake_dream_db) -> None:
    """Every pass these callbacks name has the row its orchestrator left at
    submit, carrying the usage attribution the pass started under: the one
    the callback restores and every charge goes by. A test about the row
    seeds its own over it."""
    for pass_id in ("p1", "p-expert", "p-missing"):
        _seed_submitted_pass(fake_dream_db, pass_id)


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

    async def fake_eval(script, numkeys, key, token, *argv):
        # The only Lua the dream path runs is the lock's two single-key
        # scripts, compare-and-delete and compare-and-extend (the lease
        # renewal, which passes the TTL); mirror them on the string store.
        if string_store.get(key) != token:
            return 0
        if not argv:
            string_store.pop(key, None)
        return 1

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
    *,
    pass_id: str = "p1",
    job_id: str = "j1",
    phase: str = "consolidate",
    expert_id: str | None = None,
) -> PendingEntry:
    now = datetime.now(timezone.utc)
    custom_id = f"{pass_id}:{phase}"
    payload = {
        "user_id": "u1",
        "cost_context": {"user_id": "u1", "trial_id": None, "generation": None},
        "pass_id": pass_id,
        "job_id": job_id,
        "phase": phase,
        "phase_models": {
            "consolidate": "claude-sonnet-5",
            "recombine": "claude-opus-5-5",
            "sanitize": "claude-sonnet-5",
        },
        "custom_ids": [custom_id],
        "phase_for_custom_id": {custom_id: phase},
    }
    if expert_id is not None:
        payload["expert_id"] = expert_id
    return PendingEntry(
        provider="anthropic",
        provider_batch_id=f"msgbatch_{phase}",
        callback_namespace="dream_pass",
        submitted_at=now,
        next_poll_at=now,
        payload=payload,
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


async def _persist_autopilot_bundle(pass_id: str = "p1") -> None:
    from backend.copilot.dream.batch_submit import persist_input_bundle
    from backend.copilot.dream.fetch import DreamInput

    now = datetime.now(timezone.utc)
    await persist_input_bundle(
        pass_id,
        DreamInput(
            user_id="u1",
            group_id="user_u1",
            window_start=now,
            window_end=now,
        ),
    )


class TestPhaseChaining:
    @pytest.mark.asyncio
    async def test_terminal_result_without_input_bundle_cannot_apply(
        self, fake_redis
    ) -> None:
        apply = AsyncMock()
        mark_errored = AsyncMock()
        release_lock = AsyncMock()

        with (
            patch("backend.copilot.dream.apply.apply_operations", apply),
            patch(
                "backend.copilot.dream.job_status.mark_errored",
                mark_errored,
            ),
            patch(
                "backend.copilot.dream.cleanup.release_dream_lock",
                release_lock,
            ),
        ):
            await handle_dream_batch_result(
                _entry(pass_id="p-missing", phase="sanitize"),
                [
                    _row(
                        custom_id="p-missing:sanitize",
                        content=_SANITIZE_CONTENT,
                    )
                ],
            )

        apply.assert_not_awaited()
        assert mark_errored.await_args.kwargs["error"] == "batch: input_missing"
        # No token anywhere (no bundle, no row): the pass's ownership of the
        # lock is unknown, so nothing is released; its TTL, or the reaper,
        # settles it.
        release_lock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_autopilot_payload_cannot_apply_expert_dream(
        self, fake_redis
    ) -> None:
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p-expert",
            DreamInput(
                user_id="u1",
                expert_id="expert-1",
                group_id="expert_resolved",
                window_start=now,
                window_end=now,
            ),
            lock_token="tok-expert",
        )
        apply = AsyncMock()
        mark_errored = AsyncMock()
        release_lock = AsyncMock()

        with (
            patch("backend.copilot.dream.apply.apply_operations", apply),
            patch(
                "backend.copilot.dream.job_status.mark_errored",
                mark_errored,
            ),
            patch(
                "backend.copilot.dream.cleanup.release_dream_lock",
                release_lock,
            ),
        ):
            await handle_dream_batch_result(
                _entry(pass_id="p-expert", phase="sanitize"),
                [
                    _row(
                        custom_id="p-expert:sanitize",
                        content=_SANITIZE_CONTENT,
                    )
                ],
            )

        apply.assert_not_awaited()
        assert mark_errored.await_args.kwargs["error"] == "batch: scope_mismatch"
        release_lock.assert_awaited_once_with(
            MemoryScope.for_expert("u1", "expert-1"), "tok-expert"
        )

    @pytest.mark.asyncio
    async def test_cross_expert_payload_cannot_apply_other_expert_dream(
        self, fake_redis
    ) -> None:
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p-expert",
            DreamInput(
                user_id="u1",
                expert_id="expert-1",
                group_id="expert_resolved",
                window_start=now,
                window_end=now,
            ),
            lock_token="tok-expert",
        )
        submit_phase = AsyncMock()
        mark_errored = AsyncMock()
        release_lock = AsyncMock()

        with (
            patch(
                "backend.copilot.dream.batch_callbacks.submit_phase",
                submit_phase,
            ),
            patch(
                "backend.copilot.dream.job_status.mark_errored",
                mark_errored,
            ),
            patch(
                "backend.copilot.dream.cleanup.release_dream_lock",
                release_lock,
            ),
        ):
            await handle_dream_batch_result(
                _entry(
                    pass_id="p-expert",
                    phase="consolidate",
                    expert_id="expert-2",
                ),
                [
                    _row(
                        custom_id="p-expert:consolidate",
                        content=_CONSOLIDATE_CONTENT,
                    )
                ],
            )

        submit_phase.assert_not_awaited()
        assert mark_errored.await_args.kwargs["error"] == "batch: scope_mismatch"
        release_lock.assert_awaited_once_with(
            MemoryScope.for_expert("u1", "expert-1"), "tok-expert"
        )

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

        with (
            patch("backend.copilot.dream.batch_callbacks.submit_phase", submit_phase),
            patch(
                "backend.copilot.dream.job_status.update_status_phase", update_status
            ),
            patch(
                "backend.copilot.dream.batch_callbacks.anthropic_api_key",
                return_value="sk-ant-test",
            ),
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
        # The landed phase keeps the model it ran on, to be charged on it.
        state = await read_state("p1")
        assert state["consolidate"]["model"] == "claude-sonnet-5"

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
        from backend.copilot.dream.batch_state import write_phase_to_state

        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )

        submit_phase = AsyncMock(
            return_value=MagicMock(provider_batch_id="msgbatch_sanitize")
        )
        with (
            patch("backend.copilot.dream.batch_callbacks.submit_phase", submit_phase),
            patch(
                "backend.copilot.dream.batch_callbacks.anthropic_api_key",
                return_value="sk-ant-test",
            ),
            patch("backend.copilot.dream.job_status.update_status_phase", AsyncMock()),
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
        anthropic_batch path (half the catalog list price), JobStatus
        flips to complete."""
        from backend.copilot.dream.batch_state import write_phase_to_state
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput, FactRow

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
                facts=[
                    FactRow(
                        uuid="fact-1",
                        source="A",
                        target="B",
                        name="likes",
                        fact="A likes B",
                        scope="project:bread",
                        confidence=0.7,
                        status="active",
                        created_at=None,
                    )
                ],
                known_fact_uuids={"fact-1"},
                known_episode_uuids={"episode-1"},
            ),
        )
        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        await write_phase_to_state(
            pass_id="p1",
            phase="recombine",
            row=_row(custom_id="p1:recombine", content=_RECOMBINE_CONTENT),
        )

        # Real batch-path apply reports ``skipped`` (drain not run by design).
        apply = AsyncMock(
            return_value={
                "writes": 0,
                "snapshot": "...",
                "ingestion_drain_status": IngestionDrainStatus.skipped,
            }
        )
        mark_complete = AsyncMock()
        persist = AsyncMock()
        release_lock = AsyncMock()
        with (
            patch("backend.copilot.dream.apply.apply_operations", apply),
            patch("backend.copilot.dream.job_status.mark_complete", mark_complete),
            patch("backend.copilot.inference.record.persist_and_record_usage", persist),
            patch("backend.copilot.dream.cleanup.release_dream_lock", release_lock),
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        apply.assert_awaited_once()
        # The demotion allowlist is threaded from the bundle already loaded
        # for the clamp — apply must not re-read the bundle from Redis.
        assert apply.call_args.kwargs["known_fact_uuids"] == {"fact-1"}
        assert apply.call_args.kwargs["known_episode_uuids"] == {"episode-1"}
        # ...and each fact's scope, which a write's fact citations must share.
        assert apply.call_args.kwargs["fact_scopes"] == {"fact-1": "project:bread"}
        # The batch path must NOT run the 300s in-line ingestion drain: apply
        # executes inside this handler, which BatchExecutor.walk_once awaits
        # serially — a long drain would stall every other user's batch poll.
        assert apply.call_args.kwargs["ingestion_drain_timeout"] == 0
        mark_complete.assert_awaited_once()
        # The sanitizer's user-facing narrative must ride on the result so the
        # Memory Visualizer isn't blank for batch-completed dreams.
        assert mark_complete.call_args.kwargs["result"].summary_for_user == "ok"
        # Drain was skipped by design on the batch path — the tri-state says
        # so (``skipped``), distinguishing it from a real sync-path failure
        # (``timed_out``) rather than masking either as success.
        assert (
            mark_complete.call_args.kwargs["result"].ingestion_drain_status
            is IngestionDrainStatus.skipped
        )
        # The batch path disowned the dream lock to this callback; the
        # terminal handler must release it with the ownership token the
        # input bundle carried — compare-and-delete, never a blind DEL.
        release_lock.assert_awaited_once_with(MemoryScope.for_user("u1"), "tok-u1")
        # One cost-log row per phase (consolidate, recombine, sanitize), in
        # the dream's row shape, on the Anthropic batch route.
        assert persist.await_count == 3
        rows = {
            call.kwargs["extra_metadata"]["dream_phase"]: call.kwargs
            for call in persist.await_args_list
        }
        assert list(rows) == ["consolidate", "recombine", "sanitize"]
        for phase, row in rows.items():
            assert row["provider"] == "anthropic"
            assert row["block_name_override"] == f"copilot:dream:{phase}"
            assert row["graph_exec_id_override"] == "p1"
            assert row["expert_id"] is None
            assert row["skip_daily"] is True
            assert row["extra_metadata"]["execution_path"] == "anthropic_batch"
            assert row["extra_metadata"]["discount_applied"] == 0.5
            assert row["extra_metadata"]["dream_pass_id"] == "p1"
        # Each phase is priced with ITS OWN model — recombine uses the
        # advanced (opus) model, not phase 1's standard model.
        models_by_phase = {phase: row["model"] for phase, row in rows.items()}
        assert models_by_phase["consolidate"] == "claude-sonnet-5"
        assert models_by_phase["recombine"] == "claude-opus-5-5"
        assert models_by_phase["sanitize"] == "claude-sonnet-5"
        # ...and priced from that model's catalog card at half the list
        # price: each ``_row`` carries 10 input + 20 output tokens.
        costs_by_phase = {phase: row["cost_usd"] for phase, row in rows.items()}
        sonnet_5_cost = (10 * 2.0 + 20 * 10.0) / 1_000_000 / 2
        opus_5_5_cost = (10 * 4.0 + 20 * 20.0) / 1_000_000 / 2
        assert costs_by_phase["consolidate"] == pytest.approx(sonnet_5_cost)
        assert costs_by_phase["recombine"] == pytest.approx(opus_5_5_cost)
        assert costs_by_phase["sanitize"] == pytest.approx(sonnet_5_cost)

    @pytest.mark.asyncio
    async def test_expert_terminal_result_applies_and_releases_in_expert_scope(
        self, fake_redis
    ):
        from backend.copilot.dream.batch_state import write_phase_to_state
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        _, _, string_store = fake_redis
        expert_lock = MemoryScope.for_expert("u1", "expert-1").redis_key("dream_lock")
        string_store[expert_lock] = "tok-expert"
        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p-expert",
            DreamInput(
                user_id="u1",
                expert_id="expert-1",
                group_id="expert_resolved",
                window_start=now,
                window_end=now,
            ),
            lock_token="tok-expert",
        )
        await write_phase_to_state(
            pass_id="p-expert",
            phase="consolidate",
            row=_row(custom_id="p-expert:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        await write_phase_to_state(
            pass_id="p-expert",
            phase="recombine",
            row=_row(custom_id="p-expert:recombine", content=_RECOMBINE_CONTENT),
        )

        apply = AsyncMock(
            return_value={
                "writes": 0,
                "ingestion_drain_status": IngestionDrainStatus.skipped,
            }
        )
        release_lock = AsyncMock()
        persist = AsyncMock()
        with (
            patch("backend.copilot.dream.apply.apply_operations", apply),
            patch("backend.copilot.dream.job_status.mark_complete", new=AsyncMock()),
            patch(
                "backend.copilot.inference.record.persist_and_record_usage",
                new=persist,
            ),
            patch(
                "backend.copilot.dream.cleanup.release_dream_lock",
                release_lock,
            ),
        ):
            await handle_dream_batch_result(
                _entry(pass_id="p-expert", phase="sanitize", expert_id="expert-1"),
                [_row(custom_id="p-expert:sanitize", content=_SANITIZE_CONTENT)],
            )

        assert apply.call_args.args[0] == MemoryScope.for_expert("u1", "expert-1")
        release_lock.assert_awaited_once_with(
            MemoryScope.for_expert("u1", "expert-1"), "tok-expert"
        )
        # The expert's pass is charged to its owner and attributed to it.
        assert persist.await_count == 3
        for call in persist.await_args_list:
            assert call.kwargs["user_id"] == "u1"
            assert call.kwargs["expert_id"] == "expert-1"
            assert call.kwargs["extra_metadata"]["expert_id"] == "expert-1"

    @pytest.mark.asyncio
    async def test_redispatch_after_charge_does_not_double_bill(self, fake_redis):
        """If the BatchExecutor crashes between charging and
        ``remove_pending``, the next walk re-dispatches the same batch.
        The Redis dedup gates must prevent BOTH the second charge and a
        second ``apply_operations`` run."""
        from backend.copilot.dream.batch_state import write_phase_to_state
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
        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        await write_phase_to_state(
            pass_id="p1",
            phase="recombine",
            row=_row(custom_id="p1:recombine", content=_RECOMBINE_CONTENT),
        )

        apply = AsyncMock(return_value={"writes": 0, "snapshot": "..."})
        record_cost = AsyncMock()
        with (
            patch("backend.copilot.dream.apply.apply_operations", apply),
            patch("backend.copilot.dream.job_status.mark_complete", AsyncMock()),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", record_cost),
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
        from backend.copilot.dream.batch_state import write_phase_to_state
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
        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        await write_phase_to_state(
            pass_id="p1",
            phase="recombine",
            row=_row(custom_id="p1:recombine", content=_RECOMBINE_CONTENT),
        )

        apply = AsyncMock(return_value={"writes": 0, "snapshot": "..."})
        mark_complete = AsyncMock()
        release_lock = AsyncMock()
        # Crash-before-cleanup simulation: state + input bundle survive the
        # first delivery, so the re-dispatch sees a fully populated pass.
        with (
            patch("backend.copilot.dream.apply.apply_operations", apply),
            patch("backend.copilot.dream.job_status.mark_complete", mark_complete),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()),
            patch("backend.copilot.dream.cleanup.release_dream_lock", release_lock),
            patch("backend.copilot.dream.batch_state.delete_state", AsyncMock()),
            patch("backend.copilot.dream.batch_state.delete_input_bundle", AsyncMock()),
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
            assert call.args == (MemoryScope.for_user("u1"), "tok-u1")

    @pytest.mark.asyncio
    async def test_apply_gate_redis_outage_fails_pass_not_silent_success(
        self, fake_redis
    ):
        """A Redis outage while claiming the apply gate means we cannot
        distinguish first delivery from duplicate. The pass must be marked
        errored — completing it would report success while no memory was
        written, silently dropping the dream."""
        from backend.copilot.dream.batch_state import write_phase_to_state

        await _persist_autopilot_bundle()
        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        await write_phase_to_state(
            pass_id="p1",
            phase="recombine",
            row=_row(custom_id="p1:recombine", content=_RECOMBINE_CONTENT),
        )

        apply = AsyncMock()
        mark_complete = AsyncMock()
        mark_errored = AsyncMock()
        gate = AsyncMock(return_value="error")
        with (
            patch("backend.copilot.dream.apply.apply_operations", apply),
            patch("backend.copilot.dream.job_status.mark_complete", mark_complete),
            patch("backend.copilot.dream.job_status.mark_errored", mark_errored),
            patch("backend.copilot.dream.batch_callbacks.claim_apply_gate", gate),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()),
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        apply.assert_not_awaited()
        mark_complete.assert_not_awaited()
        mark_errored.assert_awaited_once()
        assert mark_errored.call_args.kwargs["error"] == "apply: gate_unavailable"


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

        with patch("backend.copilot.dream.cleanup.release_dream_lock", AsyncMock()):
            await handle_dream_batch_result(entry, [])

        final = await read_status(kind="dream_pass", job_id="j-dead")
        assert final is not None
        assert final.state == "errored"
        assert final.error == "batch: payload_incomplete"

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

        with patch("backend.copilot.dream.cleanup.release_dream_lock", AsyncMock()):
            await handle_dream_batch_result(entry, [])

        final = await read_status(kind="dream_pass", job_id="j-odd")
        assert final is not None
        assert final.state == "errored"
        assert final.error == "batch: unknown_phase"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", ["content moderation", "max_tokens"])
    async def test_errored_row_short_circuits_to_mark_errored(self, fake_redis, error):
        await _persist_autopilot_bundle()
        mark_errored = AsyncMock()
        record_cost = AsyncMock()
        with (
            patch("backend.copilot.dream.job_status.mark_errored", mark_errored),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", record_cost),
        ):
            await handle_dream_batch_result(
                _entry(phase="consolidate"),
                [
                    _row(
                        custom_id="p1:consolidate",
                        content="",
                        error=error,
                    )
                ],
            )
        mark_errored.assert_awaited_once()
        assert mark_errored.call_args.kwargs["error"] == "consolidate: provider_error"
        record_cost.assert_not_awaited()
        redis, _, _ = fake_redis
        stored = json.loads(redis.hset.await_args.args[2])
        assert stored["error"] == error
        assert stored["input_tokens"] == 10
        assert stored["output_tokens"] == 20

    @pytest.mark.asyncio
    @pytest.mark.parametrize("content", ["this is not JSON", '{"facts": "invalid"}'])
    async def test_invalid_json_marks_errored_does_not_chain(self, fake_redis, content):
        await _persist_autopilot_bundle()
        submit_phase = AsyncMock()
        mark_errored = AsyncMock()
        record_cost = AsyncMock()
        with (
            patch("backend.copilot.dream.batch_callbacks.submit_phase", submit_phase),
            patch("backend.copilot.dream.job_status.mark_errored", mark_errored),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", record_cost),
        ):
            await handle_dream_batch_result(
                _entry(phase="consolidate"),
                [
                    _row(
                        custom_id="p1:consolidate",
                        content=content,
                    )
                ],
            )
        submit_phase.assert_not_awaited()
        mark_errored.assert_awaited_once()
        assert mark_errored.call_args.kwargs["error"] == "consolidate: schema_mismatch"
        record_cost.assert_not_awaited()
        redis, _, _ = fake_redis
        stored = json.loads(redis.hset.await_args.args[2])
        assert "invalid output shape" in stored["error"]
        assert stored["input_tokens"] == 10
        assert stored["output_tokens"] == 20

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content,error",
        [
            (_RECOMBINE_CONTENT, "max_tokens"),
            ('{"proposals": "invalid"}', None),
            ("", "provider error"),
        ],
    )
    async def test_failed_phase_only_charges_earlier_completed_phase(
        self, fake_redis, content, error
    ):
        await _persist_autopilot_bundle()
        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        record_cost = AsyncMock()
        with (
            patch("backend.copilot.dream.job_status.mark_errored", AsyncMock()),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", record_cost),
        ):
            await handle_dream_batch_result(
                _entry(phase="recombine"),
                [_row(custom_id="p1:recombine", content=content, error=error)],
            )
        record_cost.assert_awaited_once()
        ctx, usage = record_cost.await_args.args
        assert ctx.job.phase == "consolidate"
        assert (usage.input_tokens, usage.output_tokens) == (10, 20)

    @pytest.mark.asyncio
    async def test_text_answer_around_the_json_still_chains(self, fake_redis):
        """A model the output tool can't be forced on (``auto``) may answer
        in text. The JSON in it, fenced behind a line of prose, is the
        phase result, and the next phase reads the JSON alone."""
        await _persist_autopilot_bundle()
        submit_phase = AsyncMock(
            return_value=MagicMock(provider_batch_id="msgbatch_recombine")
        )
        mark_errored = AsyncMock()
        with (
            patch("backend.copilot.dream.batch_callbacks.submit_phase", submit_phase),
            patch("backend.copilot.dream.job_status.mark_errored", mark_errored),
            patch("backend.copilot.dream.job_status.update_status_phase", AsyncMock()),
            patch(
                "backend.copilot.dream.batch_callbacks.anthropic_api_key",
                return_value="sk-ant-test",
            ),
        ):
            await handle_dream_batch_result(
                _entry(phase="consolidate"),
                [
                    _row(
                        custom_id="p1:consolidate",
                        content='Here are the facts:\n```json\n{"facts": []}\n```',
                    )
                ],
            )
        mark_errored.assert_not_awaited()
        submit_phase.assert_awaited_once()
        assert submit_phase.call_args.kwargs["consolidated_json"] == (
            _CONSOLIDATE_CONTENT
        )

    @pytest.mark.asyncio
    async def test_apply_crash_marks_errored_still_records_usage(self, fake_redis):
        """If apply raises, the pass is errored — but the three LLM phases
        already ran and Anthropic billed us, so their usage is still
        recorded (matches the sync path + dream/billing.py)."""
        from backend.copilot.dream.batch_state import write_phase_to_state
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1", group_id="user_u1", window_start=now, window_end=now
            ),
        )
        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        await write_phase_to_state(
            pass_id="p1",
            phase="recombine",
            row=_row(custom_id="p1:recombine", content=_RECOMBINE_CONTENT),
        )

        apply = AsyncMock(side_effect=RuntimeError("FalkorDB unreachable"))
        mark_errored = AsyncMock()
        record_cost = AsyncMock()
        with (
            patch("backend.copilot.dream.apply.apply_operations", apply),
            patch("backend.copilot.dream.job_status.mark_errored", mark_errored),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", record_cost),
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
        """An unexpected raise OUTSIDE the handler's own fail_pass guards
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
        with (
            patch("backend.copilot.dream.batch_callbacks._chain_next_phase", chain),
            patch("backend.copilot.dream.job_status.mark_errored", mark_errored),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", record_cost),
            patch("backend.copilot.dream.cleanup.release_dream_lock", release_lock),
        ):
            # Must not propagate — the crash guard finalizes and swallows.
            await handle_dream_batch_result(
                _entry(phase="consolidate"),
                [_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT)],
            )

        mark_errored.assert_awaited_once()
        assert mark_errored.call_args.kwargs["error"] == "consolidate: crashed"
        # Released with the ownership token the input bundle carried.
        release_lock.assert_awaited_once_with(MemoryScope.for_user("u1"), "tok-u1")


class TestMalformedPayload:
    @pytest.mark.asyncio
    async def test_missing_pass_id_leaves_the_lock_to_its_ttl(self, fake_redis):
        """Malformed payload (missing pass_id) early-returns: with no pass_id
        there is no persisted token to release the lock with and no row to
        mark for the reaper, so the lock is left to its TTL rather than
        blind-deleted."""
        _, _, string_store = fake_redis
        string_store["dream:inflight:u1"] = "tok-u1"
        entry = _entry()
        entry.payload["pass_id"] = ""
        await handle_dream_batch_result(
            entry,
            [_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT)],
        )
        assert string_store["dream:inflight:u1"] == "tok-u1"

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
        with patch("backend.copilot.dream.cleanup.release_dream_lock", release):
            await handle_dream_batch_result(entry, [_row(custom_id="x", content="y")])
        release.assert_awaited_once_with(MemoryScope.for_user("u1"), "tok-u1")


class TestLockTokenWiring:
    """End-to-end token flow with the real ``release_dream_lock`` — the
    fake redis implements the single-key compare-and-delete Lua."""

    async def _seed_terminal_pass(self) -> None:
        from backend.copilot.dream.batch_state import write_phase_to_state
        from backend.copilot.dream.batch_submit import persist_input_bundle
        from backend.copilot.dream.fetch import DreamInput

        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1", group_id="user_u1", window_start=now, window_end=now
            ),
        )
        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        await write_phase_to_state(
            pass_id="p1",
            phase="recombine",
            row=_row(custom_id="p1:recombine", content=_RECOMBINE_CONTENT),
        )

    async def _dispatch_sanitize(self) -> None:
        with (
            patch(
                "backend.copilot.dream.apply.apply_operations",
                AsyncMock(return_value={"writes": 0}),
            ),
            patch("backend.copilot.dream.job_status.mark_complete", AsyncMock()),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()),
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
        (no token: the lock is left, never blind-deleted) — letting it
        propagate would hit the handler's crash guard, whose fail_pass
        rewrites the already-completed job to errored."""
        _, _, string_store = fake_redis
        string_store["dream:inflight:u1"] = "tok-u1"
        await self._seed_terminal_pass()

        mark_complete = AsyncMock()
        mark_errored = AsyncMock()
        release_lock = AsyncMock()

        async def blips_once_the_job_is_complete(pass_id: str) -> str:
            if mark_complete.await_count:
                raise ConnectionError("redis blip")
            return "tok-u1"

        read_token = AsyncMock(side_effect=blips_once_the_job_is_complete)
        with (
            patch(
                "backend.copilot.dream.apply.apply_operations",
                AsyncMock(return_value={"writes": 0}),
            ),
            patch("backend.copilot.dream.job_status.mark_complete", mark_complete),
            patch("backend.copilot.dream.job_status.mark_errored", mark_errored),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()),
            patch("backend.copilot.dream.batch_outcome.read_lock_token", read_token),
            patch("backend.copilot.dream.cleanup.release_dream_lock", release_lock),
            # The row never got the token: the lease renewal's write failed.
            patch(
                "backend.copilot.dream.lease.record_lease",
                AsyncMock(return_value=None),
            ),
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        mark_complete.assert_awaited_once()
        mark_errored.assert_not_awaited()
        # The token read failed and the row keeps none either: ownership is
        # unknown, so nothing is released rather than blind-deleted.
        release_lock.assert_not_awaited()
        assert string_store["dream:inflight:u1"] == "tok-u1"

    @pytest.mark.asyncio
    async def test_cleanup_failure_after_complete_keeps_job_completed(self, fake_redis):
        """A Redis blip on the post-mark_complete state/bundle deletes must
        not route through the crash guard to fail_pass — both keys carry
        24h TTLs, so cleanup is best-effort and the completed job stays
        completed."""
        _, _, string_store = fake_redis
        string_store["dream:inflight:u1"] = "tok-u1"
        await self._seed_terminal_pass()

        mark_complete = AsyncMock()
        mark_errored = AsyncMock()
        delete_state = AsyncMock(side_effect=ConnectionError("redis blip"))
        with (
            patch(
                "backend.copilot.dream.apply.apply_operations",
                AsyncMock(return_value={"writes": 0}),
            ),
            patch("backend.copilot.dream.job_status.mark_complete", mark_complete),
            patch("backend.copilot.dream.job_status.mark_errored", mark_errored),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()),
            patch("backend.copilot.dream.batch_state.delete_state", delete_state),
            patch("backend.copilot.dream.cleanup.release_dream_lock", AsyncMock()),
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
        with (
            patch("backend.copilot.dream.apply.apply_operations", apply),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()),
            patch("backend.copilot.dream.cleanup.release_dream_lock", AsyncMock()),
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
        # The batch path never drains in-line, so the finalized result marks
        # the writes as a by-design skip (``skipped``), not a drain failure.
        assert final.result["ingestion_drain_status"] == IngestionDrainStatus.skipped

    @pytest.mark.asyncio
    async def test_duplicate_dispatch_leaves_terminal_job_untouched(self, fake_redis):
        """A duplicate against an already-completed row must not rewrite it
        — the first delivery's real apply stats stay authoritative."""
        _, _, string_store = fake_redis
        await self._seed_terminal_pass()
        string_store["dream:applied:p1"] = "1"

        mark_complete = AsyncMock()
        read_existing = AsyncMock(return_value=MagicMock(state="complete"))
        with (
            patch("backend.copilot.dream.apply.apply_operations", AsyncMock()),
            patch("backend.copilot.dream.job_status.mark_complete", mark_complete),
            patch("backend.copilot.dream.job_status.read_status", read_existing),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()),
            patch("backend.copilot.dream.cleanup.release_dream_lock", AsyncMock()),
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


def _seed_submitted_pass(fake_dream_db, pass_id: str = "p1") -> None:
    """The row as the orchestrator leaves it once consolidate is submitted."""
    fake_dream_db.seed(
        DreamPassDraft(
            id=pass_id,
            user_id="u1",
            scope_key="u1",
            route=DreamPassRoute.ANTHROPIC_BATCH,
            trigger=DreamPassTrigger.CRON,
            status=DreamPassStatus.SUBMITTED,
            phase=DreamPassPhase.CONSOLIDATE,
            started_at=datetime.now(timezone.utc),
        )
    )


class TestDreamPassRecord:
    """Each callback advances the pass's durable row: the phase that landed
    and its output, the next batch, the apply, and the end with what the
    landed phases used."""

    @pytest.mark.asyncio
    async def test_the_batch_chain_records_every_step(self, fake_redis, fake_dream_db):
        _, _, string_store = fake_redis
        string_store["dream:inflight:u1"] = "tok-u1"
        await _persist_autopilot_bundle()
        _seed_submitted_pass(fake_dream_db)
        submit_phase = AsyncMock(
            side_effect=[
                MagicMock(provider_batch_id="msgbatch_recombine"),
                MagicMock(provider_batch_id="msgbatch_sanitize"),
            ]
        )
        apply = AsyncMock(
            return_value={
                "session_id": "s1",
                "consolidated_count": 2,
                "proposal_count": 1,
                "demotion_count": 0,
                "entity_invalidation_count": 0,
                "ingestion_drain_status": IngestionDrainStatus.skipped,
                "snapshot": DreamOperationsSnapshot(),
            }
        )
        with (
            patch("backend.copilot.dream.batch_callbacks.submit_phase", submit_phase),
            patch(
                "backend.copilot.dream.batch_callbacks.anthropic_api_key",
                return_value="sk-ant-test",
            ),
            patch("backend.copilot.dream.job_status.update_status_phase", AsyncMock()),
            patch("backend.copilot.dream.apply.apply_operations", apply),
            patch("backend.copilot.dream.job_status.mark_complete", AsyncMock()),
            patch(
                "backend.copilot.inference.record.persist_and_record_usage", AsyncMock()
            ),
            patch("backend.copilot.dream.cleanup.release_dream_lock", AsyncMock()),
        ):
            for phase, content in (
                ("consolidate", _CONSOLIDATE_CONTENT),
                ("recombine", _RECOMBINE_CONTENT),
                ("sanitize", _SANITIZE_CONTENT),
            ):
                await handle_dream_batch_result(
                    _entry(phase=phase),
                    [_row(custom_id=f"p1:{phase}", content=content)],
                )

        assert fake_dream_db.phases("p1") == [
            DreamPassPhase.RECOMBINE,
            DreamPassPhase.SANITIZE,
            DreamPassPhase.APPLY,
            DreamPassPhase.DONE,
        ]
        assert fake_dream_db.statuses("p1") == [
            DreamPassStatus.APPLYING,
            DreamPassStatus.COMPLETE,
        ]
        assert [
            update.provider_batch_id
            for _, update in fake_dream_db.writes
            if update.provider_batch_id
        ] == ["msgbatch_recombine", "msgbatch_sanitize"]
        row = fake_dream_db.rows["p1"]
        assert list(row["phase_outputs"]) == ["consolidate", "recombine", "sanitize"]
        assert row["operations"]["planned"] == DreamOperations(summary_for_user="ok")
        applied = row["operations"]["applied"]
        assert (applied.consolidated_count, applied.dream_session_id) == (2, "s1")
        # Each landed phase priced on its own model at the batch discount,
        # as its cost row is: 10 input + 20 output tokens per ``_row``.
        usage = row["usage"]
        assert [p.model for p in usage.phases] == [
            "claude-sonnet-5",
            "claude-opus-5-5",
            "claude-sonnet-5",
        ]
        sonnet_5_cost = (10 * 2.0 + 20 * 10.0) / 1_000_000 / 2
        opus_5_5_cost = (10 * 4.0 + 20 * 20.0) / 1_000_000 / 2
        assert usage.total_cost_usd == pytest.approx(2 * sonnet_5_cost + opus_5_5_cost)
        assert usage.discount_applied == 0.5
        # What the eval driver reads back: a batch pass's result with usage.
        result = dream_pass_result_from_row(fake_dream_db.record("p1"))
        assert result.execution_path == "anthropic_batch"
        assert result.usage == usage
        assert (result.consolidated_count, result.proposal_count) == (2, 1)
        assert result.ingestion_drain_status is IngestionDrainStatus.skipped

    @pytest.mark.asyncio
    async def test_a_failed_phase_closes_the_row_with_the_landed_usage(
        self, fake_redis, fake_dream_db
    ):
        await _persist_autopilot_bundle()
        _seed_submitted_pass(fake_dream_db)
        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        with (
            patch("backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()),
            patch("backend.copilot.dream.cleanup.release_dream_lock", AsyncMock()),
        ):
            await handle_dream_batch_result(
                _entry(phase="recombine"),
                [_row(custom_id="p1:recombine", content="", error="provider down")],
            )

        assert fake_dream_db.statuses("p1") == [DreamPassStatus.ERRORED]
        row = fake_dream_db.rows["p1"]
        assert row["error"] == "recombine: provider_error"
        # The errored phase used nothing billable; consolidate did.
        assert [p.phase for p in row["usage"].phases] == ["consolidate"]
        assert row["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_an_apply_crash_closes_the_row_errored_after_applying(
        self, fake_redis, fake_dream_db
    ):
        _, _, string_store = fake_redis
        string_store["dream:inflight:u1"] = "tok-u1"
        await _persist_autopilot_bundle()
        _seed_submitted_pass(fake_dream_db)
        with (
            patch(
                "backend.copilot.dream.apply.apply_operations",
                AsyncMock(side_effect=RuntimeError("FalkorDB unreachable")),
            ),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()),
            patch("backend.copilot.dream.cleanup.release_dream_lock", AsyncMock()),
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        assert fake_dream_db.statuses("p1") == [
            DreamPassStatus.APPLYING,
            DreamPassStatus.ERRORED,
        ]
        row = fake_dream_db.rows["p1"]
        assert row["error"] == "apply: crashed", "no exception text is recorded"
        assert row["operations"]["planned"] == DreamOperations(summary_for_user="ok")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload_fix, error",
        [
            ({"phase": "daydream"}, "batch: unknown_phase"),
            ({"phase": None}, "batch: payload_incomplete"),
        ],
        ids=["unknown_phase", "missing_phase"],
    )
    async def test_a_dead_end_payload_closes_the_row_with_landed_usage(
        self, fake_redis, fake_dream_db, payload_fix, error
    ):
        """The consolidate phase landed and was billed; a payload no phase
        handler can take then ends the pass. The record is closed with that
        phase's usage before the disowned lock is released, under the token
        the row keeps: the pass never persisted a bundle to carry one."""
        _seed_submitted_pass(fake_dream_db)
        fake_dream_db.rows["p1"]["lease_token"] = "tok-u1"
        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        entry = _entry(phase="recombine")
        entry.payload.update(payload_fix)
        order: list[str] = []
        record = fake_dream_db.update_dream_pass

        async def recorded(pass_id, update):
            order.append(f"record {update.status}")
            return await record(pass_id, update)

        release_lock = AsyncMock(side_effect=lambda *_: order.append("lock released"))
        charge = AsyncMock(side_effect=lambda ctx, usage: order.append("charged"))
        with (
            patch.object(fake_dream_db, "update_dream_pass", recorded),
            patch("backend.copilot.dream.cleanup.release_dream_lock", release_lock),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", charge),
        ):
            await handle_dream_batch_result(entry, [])

        row = fake_dream_db.rows["p1"]
        assert row["status"] is DreamPassStatus.ERRORED
        assert row["error"] == error
        assert [p.phase for p in row["usage"].phases] == ["consolidate"]
        # Like any failure: the landed phase charged once, and the pass's
        # batch state gone with its lock.
        assert order == ["record ERRORED", "charged", "lock released"]
        assert release_lock.await_args.args == (MemoryScope.for_user("u1"), "tok-u1")
        assert state_key("p1") not in fake_redis[1]

    @pytest.mark.asyncio
    async def test_a_dead_end_payload_naming_another_owner_ends_the_pass_as_its_rows(
        self, fake_redis, fake_dream_db
    ):
        """The payload is the part that went wrong: the landed phase is
        charged to the owner the row names, and that owner's lock released,
        never the payload's."""
        _seed_submitted_pass(fake_dream_db)
        fake_dream_db.rows["p1"]["lease_token"] = "tok-u1"
        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        entry = _entry(phase="recombine", expert_id="e-other")
        entry.payload.update({"phase": "daydream", "user_id": "someone-else"})
        release_lock = AsyncMock(return_value=True)
        charge = AsyncMock()
        with (
            patch("backend.copilot.dream.cleanup.release_dream_lock", release_lock),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", charge),
        ):
            await handle_dream_batch_result(entry, [])

        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.ERRORED
        charged_scope = charge.await_args.args[0].scope
        assert (charged_scope.user_id, charged_scope.expert_id) == ("u1", None)
        assert release_lock.await_args.args == (MemoryScope.for_user("u1"), "tok-u1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "row_read",
        [
            AsyncMock(side_effect=TimeoutError("dream pass database stalled")),
            AsyncMock(return_value=None),
        ],
        ids=["row_unreadable", "row_missing"],
    )
    async def test_a_dead_end_payload_without_its_row_ends_as_its_bundles_owner(
        self, fake_redis, fake_dream_db, row_read
    ):
        """No row to say whose pass it is: the input bundle says, as it does
        for the phase handler. The landed phase is charged to, and the lock
        released for, the bundle's owner, never the payload's."""
        _seed_submitted_pass(fake_dream_db)
        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1", group_id="user_u1", window_start=now, window_end=now
            ),
            lock_token="tok-u1",
        )
        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        entry = _entry(phase="recombine")
        entry.payload.update({"phase": "daydream", "user_id": "someone-else"})
        release_lock = AsyncMock(return_value=True)
        charge = AsyncMock()
        with (
            patch("backend.copilot.dream.batch_deliveries.read_pass", row_read),
            patch("backend.copilot.dream.cleanup.release_dream_lock", release_lock),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", charge),
        ):
            await handle_dream_batch_result(entry, [])

        assert charge.await_args.args[0].scope.user_id == "u1"
        assert release_lock.await_args.args == (MemoryScope.for_user("u1"), "tok-u1")

    @pytest.mark.asyncio
    async def test_a_dead_end_payload_with_no_record_of_its_owner_is_left_to_the_reaper(
        self, fake_redis, fake_dream_db
    ):
        """Neither the row nor the input bundle can be read: nothing is charged,
        released or deleted on the payload's word. The row closes marked, the
        landed phase kept in the state, for the reaper to clean up after the
        pass as the owner the row names."""
        _seed_submitted_pass(fake_dream_db)
        await write_phase_to_state(
            pass_id="p1",
            phase="consolidate",
            row=_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT),
        )
        entry = _entry(phase="recombine")
        entry.payload.update({"phase": "daydream", "user_id": "someone-else"})
        release_lock = AsyncMock(return_value=True)
        charge = AsyncMock()
        with (
            patch(
                "backend.copilot.dream.batch_deliveries.read_pass",
                AsyncMock(side_effect=TimeoutError("dream pass database stalled")),
            ),
            patch(
                "backend.copilot.dream.batch_deliveries.read_input_bundle",
                AsyncMock(side_effect=ConnectionError("redis down")),
            ),
            patch("backend.copilot.dream.cleanup.release_dream_lock", release_lock),
            patch("backend.copilot.dream.batch_costs.record_phase_cost", charge),
        ):
            await handle_dream_batch_result(entry, [])

        charge.assert_not_awaited()
        release_lock.assert_not_awaited()
        row = fake_dream_db.rows["p1"]
        assert row["status"] is DreamPassStatus.ERRORED
        assert row["cleanup_pending_at"] is not None
        assert state_key("p1") in fake_redis[1]

    @pytest.mark.asyncio
    async def test_an_unreadable_state_still_closes_the_row_errored(
        self, fake_dream_db, fake_dream_redis, monkeypatch
    ):
        """Only reading the pass's Redis state fails; the database and the
        rest of Redis are healthy. The crash it causes still closes the row
        ERRORED, its usage unknown, errors the admin job and releases the
        lock."""
        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1", group_id="user_u1", window_start=now, window_end=now
            ),
            lock_token="tok-u1",
        )
        lock_key = MemoryScope.for_user("u1").redis_key("dream_lock")
        fake_dream_redis.store[lock_key] = "tok-u1"
        await job_status.write_initial_status(
            kind="dream_pass", job_id="j1", user_id="u1"
        )
        _seed_submitted_pass(fake_dream_db)
        hgetall = fake_dream_redis.hgetall

        async def state_unreadable(name):
            if name.startswith("dream:batch:state:"):
                raise ConnectionError("scripted Redis state outage")
            return await hgetall(name)

        monkeypatch.setattr(fake_dream_redis, "hgetall", state_unreadable)
        submit_phase = AsyncMock()

        with (
            patch("backend.copilot.dream.batch_callbacks.submit_phase", submit_phase),
            patch(
                "backend.copilot.dream.batch_callbacks.anthropic_api_key",
                return_value="sk-ant-test",
            ),
        ):
            await handle_dream_batch_result(
                _entry(phase="consolidate"),
                [_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT)],
            )

        submit_phase.assert_not_awaited()
        row = fake_dream_db.rows["p1"]
        assert row["status"] is DreamPassStatus.ERRORED
        assert row["error"] == "consolidate: crashed"
        assert row.get("usage") is None
        status = await job_status.read_status(kind="dream_pass", job_id="j1")
        assert status is not None and status.state == "errored"
        assert lock_key not in fake_dream_redis.store

    @pytest.mark.asyncio
    async def test_applying_is_recorded_before_the_apply_gate_is_claimed(
        self, fake_dream_db, fake_dream_redis
    ):
        """A delivery that dies while its APPLYING write is in flight has not
        claimed the apply gate yet, so the next delivery still applies."""
        fake_dream_redis.store[MemoryScope.for_user("u1").redis_key("dream_lock")] = (
            "tok-u1"
        )
        await _persist_autopilot_bundle()
        _seed_submitted_pass(fake_dream_db)
        for phase, content in (
            ("consolidate", _CONSOLIDATE_CONTENT),
            ("recombine", _RECOMBINE_CONTENT),
        ):
            await write_phase_to_state(
                pass_id="p1",
                phase=phase,
                row=_row(custom_id=f"p1:{phase}", content=content),
            )
        apply = AsyncMock(return_value={"consolidated_count": 0})
        entered, release = asyncio.Event(), asyncio.Event()
        record = fake_dream_db.update_dream_pass

        async def applying_held(pass_id, update):
            if update.status is DreamPassStatus.APPLYING:
                entered.set()
                await release.wait()
            return await record(pass_id, update)

        with (
            patch("backend.copilot.dream.apply.apply_operations", apply),
            patch("backend.copilot.dream.job_status.mark_complete", AsyncMock()),
            patch(
                "backend.copilot.inference.record.persist_and_record_usage", AsyncMock()
            ),
            patch("backend.copilot.dream.cleanup.release_dream_lock", AsyncMock()),
        ):
            with patch.object(fake_dream_db, "update_dream_pass", applying_held):
                first = asyncio.create_task(
                    handle_dream_batch_result(
                        _entry(phase="sanitize"),
                        [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
                    )
                )
                await asyncio.wait_for(entered.wait(), 5)
                assert "dream:applied:p1" not in fake_dream_redis.store
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            apply.assert_not_awaited()

            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        apply.assert_awaited_once()
        assert fake_dream_redis.store["dream:applied:p1"] == "1"
        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_the_batch_walker_keeps_moving_past_a_stalled_store(
        self, stalled_dream_db
    ):
        """The executor walks its queue serially, awaiting each callback. With
        the DatabaseManager stalled, each callback still finishes (each record
        write abandoned at its deadline), so every entry is dispatched."""
        for pass_id in ("p1", "p2"):
            await _persist_autopilot_bundle(pass_id)
            await enqueue_pending(
                _entry(pass_id=pass_id, phase="consolidate").model_copy(
                    update={"provider_batch_id": f"batch-{pass_id}"}
                )
            )
        register_handler("dream_pass", handle_dream_batch_result)
        submit_phase = AsyncMock(
            side_effect=[
                MagicMock(provider_batch_id="r1"),
                MagicMock(provider_batch_id="r2"),
            ]
        )

        async def results(provider, provider_batch_id, api_key):
            pass_id = provider_batch_id.removeprefix("batch-")
            return [
                _row(custom_id=f"{pass_id}:consolidate", content=_CONSOLIDATE_CONTENT)
            ]

        with (
            patch(
                "backend.executor.batch_executor.poll_batch",
                AsyncMock(return_value="ended"),
            ),
            patch(
                "backend.executor.batch_executor.download_batch_results",
                AsyncMock(side_effect=results),
            ),
            patch(
                "backend.executor.batch_executor._claim_dispatch",
                AsyncMock(return_value=True),
            ),
            patch("backend.copilot.dream.batch_callbacks.submit_phase", submit_phase),
            patch(
                "backend.copilot.dream.batch_callbacks.anthropic_api_key",
                return_value="sk-ant-test",
            ),
            patch("backend.copilot.dream.job_status.update_status_phase", AsyncMock()),
        ):
            await asyncio.wait_for(walk_once(api_key_for=lambda _: "sk-ant-test"), 10)

        assert submit_phase.await_count == 2
        assert {call.kwargs["pass_id"] for call in submit_phase.await_args_list} == {
            "p1",
            "p2",
        }
        # Each callback's read of its usage attribution, its two writes (its
        # output, its next batch), its stop check's read and the read of its
        # row for the lock token its bundle lacks, each abandoned at the
        # deadline; the reads that got no answer let the chain go on, the
        # attribution's under the batch's copy.
        assert (stalled_dream_db.started, stalled_dream_db.cancelled) == (10, 10)

    @pytest.mark.asyncio
    async def test_a_store_outage_never_fails_a_batch_pass(
        self, fake_redis, fake_dream_db
    ):
        _, _, string_store = fake_redis
        string_store["dream:inflight:u1"] = "tok-u1"
        await _persist_autopilot_bundle()
        _seed_submitted_pass(fake_dream_db)
        fake_dream_db.fail = True
        mark_complete = AsyncMock()
        mark_errored = AsyncMock()
        release_lock = AsyncMock()
        with (
            patch(
                "backend.copilot.dream.apply.apply_operations",
                AsyncMock(return_value={"consolidated_count": 0}),
            ),
            patch("backend.copilot.dream.job_status.mark_complete", mark_complete),
            patch("backend.copilot.dream.job_status.mark_errored", mark_errored),
            patch(
                "backend.copilot.inference.record.persist_and_record_usage", AsyncMock()
            ),
            patch("backend.copilot.dream.cleanup.release_dream_lock", release_lock),
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        mark_complete.assert_awaited_once()
        mark_errored.assert_not_awaited()
        release_lock.assert_awaited_once()
        assert fake_dream_db.writes == []


async def _cancel_p1() -> str:
    assert (await cancel_dream_pass("p1", user_id="u1", reason="testing")).cancelled
    return "cancelled: by_admin"


async def _expire_p1() -> str:
    assert await write_stop(
        "p1", expired("expired: lease_lapsed", not_updated_since=None)
    )
    return "expired: lease_lapsed"


class TestAStoppedPass:
    """A pass whose row was cancelled, or expired by a newer pass's guard,
    while its batch was in flight: the callback that lands next ends it before
    it chains the next phase or claims the apply gate, through ``fail_pass``,
    after asking the provider to cancel the batch the row names. (A cancel
    asks for that itself as it closes the row; each test counts only the
    callback's own request.)"""

    @pytest.fixture
    def anthropic(self):
        """The Anthropic client the provider cancel builds."""
        client = MagicMock()
        client.messages.batches.cancel = AsyncMock()
        with (
            patch(
                "backend.copilot.dream.provider_batch.anthropic_api_key",
                return_value="sk-ant-test",
            ),
            patch(
                "backend.util.llm.providers.anthropic.AsyncAnthropic",
                return_value=client,
            ),
        ):
            yield client

    @pytest.fixture
    def charges(self):
        charge = AsyncMock()
        with patch("backend.copilot.dream.batch_costs.record_phase_cost", charge):
            yield charge

    async def _in_flight(
        self, fake_dream_db, fake_dream_redis, landed: tuple[DreamPhase, ...]
    ) -> str:
        """A batch pass on batch ``msgbatch_live``: its bundle with the lock
        token, the lock it holds, its admin job row and the phases that
        landed before this callback. Returns the lock's key."""
        now = datetime.now(timezone.utc)
        await persist_input_bundle(
            "p1",
            DreamInput(
                user_id="u1", group_id="user_u1", window_start=now, window_end=now
            ),
            lock_token="tok-u1",
        )
        lock_key = MemoryScope.for_user("u1").redis_key("dream_lock")
        fake_dream_redis.store[lock_key] = "tok-u1"
        await job_status.write_initial_status(
            kind="dream_pass", job_id="j1", user_id="u1"
        )
        contents = {
            "consolidate": _CONSOLIDATE_CONTENT,
            "recombine": _RECOMBINE_CONTENT,
        }
        for phase in landed:
            await write_phase_to_state(
                pass_id="p1",
                phase=phase,
                row=_row(custom_id=f"p1:{phase}", content=contents[phase]),
            )
        fake_dream_db.seed(
            DreamPassDraft(
                id="p1",
                user_id="u1",
                scope_key="u1",
                route=DreamPassRoute.ANTHROPIC_BATCH,
                trigger=DreamPassTrigger.CRON,
                status=DreamPassStatus.SUBMITTED,
                phase=DreamPassPhase.CONSOLIDATE,
            ),
            provider_batch_id="msgbatch_live",
        )
        return lock_key

    async def _assert_ended(
        self, fake_dream_db, fake_dream_redis, lock_key: str, error: str
    ) -> None:
        """Ended as ``fail_pass`` ends a pass: the job errored with the stop,
        the lock released, the batch state and bundle cleaned, and the row
        left closed as the stop closed it."""
        status = await job_status.read_status(kind="dream_pass", job_id="j1")
        assert status is not None and (status.state, status.error) == (
            "errored",
            error,
        )
        assert lock_key not in fake_dream_redis.store
        assert state_key("p1") not in fake_dream_redis.hashes
        assert input_bundle_key("p1") not in fake_dream_redis.store
        assert fake_dream_db.rows["p1"]["cancel_generation"] == 1
        assert DreamPassStatus.ERRORED not in fake_dream_db.statuses("p1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stop", [_cancel_p1, _expire_p1], ids=["cancelled", "expired"]
    )
    async def test_it_does_not_chain_its_next_phase(
        self,
        fake_dream_db,
        fake_dream_redis,
        anthropic,
        charges,
        stop: Callable[[], Awaitable[str]],
    ):
        lock_key = await self._in_flight(fake_dream_db, fake_dream_redis, landed=())
        error = await stop()
        anthropic.messages.batches.cancel.reset_mock()
        submit_phase = AsyncMock()

        with (
            patch("backend.copilot.dream.batch_callbacks.submit_phase", submit_phase),
            patch(
                "backend.copilot.dream.batch_callbacks.anthropic_api_key",
                return_value="sk-ant-test",
            ),
        ):
            await handle_dream_batch_result(
                _entry(phase="consolidate"),
                [_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT)],
            )

        submit_phase.assert_not_awaited()
        anthropic.messages.batches.cancel.assert_awaited_once_with("msgbatch_live")
        # The phase that landed was billed at the provider, so it is charged.
        assert [call.args[0].job.phase for call in charges.await_args_list] == [
            "consolidate"
        ]
        await self._assert_ended(fake_dream_db, fake_dream_redis, lock_key, error)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "row_attribution", [True, False], ids=["row_kept_it", "row_kept_none"]
    )
    async def test_an_unattributed_callback_ends_it_charging_only_by_its_row(
        self, fake_dream_db, fake_dream_redis, anthropic, charges, row_attribution
    ):
        """A cancelled pass whose next callback carries no usage attribution
        (#15208) ends as any failed pass ends, through ``fail_pass``. The
        phase that landed is charged once under the attribution the pass's
        row carries, the one it started under, and not at all when the row
        carries none. The batch the result came from is stopped too (the
        cancel stopped the one the row named), and the row stays as the
        cancel closed it."""
        lock_key = await self._in_flight(
            fake_dream_db, fake_dream_redis, landed=("consolidate",)
        )
        if not row_attribution:
            fake_dream_db.rows["p1"]["cost_context"] = None
        await _cancel_p1()
        entry = _entry(phase="recombine")
        del entry.payload["cost_context"]

        with pytest.raises(UsageActivationUnavailable):
            await handle_dream_batch_result(
                entry, [_row(custom_id="p1:recombine", content=_RECOMBINE_CONTENT)]
            )

        charged = [call.args[0].job.phase for call in charges.await_args_list]
        assert charged == (["consolidate"] if row_attribution else [])
        assert [
            c.args[0] for c in anthropic.messages.batches.cancel.await_args_list
        ] == [
            "msgbatch_live",
            "msgbatch_recombine",
        ]
        await self._assert_ended(
            fake_dream_db,
            fake_dream_redis,
            lock_key,
            "batch: usage_attribution_unavailable",
        )
        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_a_replayed_callback_keeps_the_job_it_ended(
        self, fake_dream_db, fake_dream_redis, anthropic, charges
    ):
        """The same consolidate batch delivered again after the stop ended the
        pass and dropped its bundle: the job keeps the stop's error, nothing
        is charged or cancelled twice."""
        await self._in_flight(fake_dream_db, fake_dream_redis, landed=())
        error = await _cancel_p1()
        anthropic.messages.batches.cancel.reset_mock()
        entry = _entry(phase="consolidate")
        rows = [_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT)]
        with (
            patch("backend.copilot.dream.batch_callbacks.submit_phase", AsyncMock()),
            patch(
                "backend.copilot.dream.batch_callbacks.anthropic_api_key",
                return_value="sk-ant-test",
            ),
        ):
            await handle_dream_batch_result(entry, rows)
            await handle_dream_batch_result(entry, rows)

        status = await job_status.read_status(kind="dream_pass", job_id="j1")
        assert status is not None and (status.state, status.error) == (
            "errored",
            error,
        )
        anthropic.messages.batches.cancel.assert_awaited_once_with("msgbatch_live")
        assert [call.args[0].job.phase for call in charges.await_args_list] == [
            "consolidate"
        ]
        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_it_does_not_claim_the_apply_gate(
        self, fake_dream_db, fake_dream_redis, anthropic, charges
    ):
        lock_key = await self._in_flight(
            fake_dream_db, fake_dream_redis, landed=("consolidate", "recombine")
        )
        error = await _cancel_p1()
        anthropic.messages.batches.cancel.reset_mock()
        apply = AsyncMock()

        with patch("backend.copilot.dream.apply.apply_operations", apply):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        apply.assert_not_awaited()
        assert "dream:applied:p1" not in fake_dream_redis.store
        anthropic.messages.batches.cancel.assert_awaited_once_with("msgbatch_live")
        assert [call.args[0].job.phase for call in charges.await_args_list] == [
            "consolidate",
            "recombine",
            "sanitize",
        ]
        await self._assert_ended(fake_dream_db, fake_dream_redis, lock_key, error)

    @pytest.mark.asyncio
    async def test_a_cancel_landing_as_applying_is_recorded_is_caught_before_the_gate(
        self, fake_dream_db, fake_dream_redis, anthropic, charges
    ):
        """The check reads the row after the APPLYING write, so a cancel that
        lands while that write is in flight still stops the pass."""
        lock_key = await self._in_flight(
            fake_dream_db, fake_dream_redis, landed=("consolidate", "recombine")
        )
        record = fake_dream_db.update_dream_pass

        async def cancelled_while_applying(pass_id, update):
            written = await record(pass_id, update)
            if update.status is DreamPassStatus.APPLYING:
                await _cancel_p1()
            return written

        apply = AsyncMock()
        with (
            patch.object(fake_dream_db, "update_dream_pass", cancelled_while_applying),
            patch("backend.copilot.dream.apply.apply_operations", apply),
        ):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

        apply.assert_not_awaited()
        assert "dream:applied:p1" not in fake_dream_redis.store
        assert fake_dream_db.statuses("p1") == [
            DreamPassStatus.APPLYING,
            DreamPassStatus.CANCELLED,
        ]
        await self._assert_ended(
            fake_dream_db, fake_dream_redis, lock_key, "cancelled: by_admin"
        )

    @pytest.mark.asyncio
    async def test_a_provider_cancel_that_fails_is_logged_and_the_pass_ends(
        self, fake_dream_db, fake_dream_redis, anthropic, charges, caplog
    ):
        lock_key = await self._in_flight(fake_dream_db, fake_dream_redis, landed=())
        error = await _cancel_p1()
        anthropic.messages.batches.cancel.reset_mock()
        anthropic.messages.batches.cancel.side_effect = RuntimeError(
            "batch has already ended"
        )
        submit_phase = AsyncMock()

        with (
            caplog.at_level(logging.WARNING),
            patch("backend.copilot.dream.batch_callbacks.submit_phase", submit_phase),
        ):
            await handle_dream_batch_result(
                _entry(phase="consolidate"),
                [_row(custom_id="p1:consolidate", content=_CONSOLIDATE_CONTENT)],
            )

        anthropic.messages.batches.cancel.assert_awaited_once_with("msgbatch_live")
        assert "did not cancel dream batch msgbatch_live" in caplog.text
        submit_phase.assert_not_awaited()
        await self._assert_ended(fake_dream_db, fake_dream_redis, lock_key, error)


@pytest.mark.asyncio
async def test_callback_restores_original_trial_generation_for_late_result(
    fake_dream_db,
):
    entry = _entry()
    entry.payload["cost_context"] = {
        "user_id": "u1",
        "trial_id": "trial-original",
        "generation": None,
    }
    # The pass's row, which the callback restores from, carries the same.
    fake_dream_db.rows["p1"]["cost_context"] = dict(entry.payload["cost_context"])
    seen = []

    async def handle(_entry, _rows):
        seen.append(get_trial_cost_context("u1"))

    with patch(
        "backend.copilot.dream.batch_callbacks._handle_dream_batch_result", handle
    ):
        await handle_dream_batch_result(entry, [])
    assert seen[0].trial_id == "trial-original"
    assert seen[0].generation is None
    assert get_trial_cost_context("u1") is None


@pytest.mark.asyncio
async def test_callback_refuses_missing_or_foreign_attribution(fake_redis):
    entry = _entry()
    del entry.payload["cost_context"]
    with pytest.raises(UsageActivationUnavailable):
        await handle_dream_batch_result(entry, [])
    entry.payload["cost_context"] = {"user_id": "foreign", "trial_id": None}
    with pytest.raises(ValueError, match="different user"):
        await handle_dream_batch_result(entry, [])


@pytest.mark.asyncio
@pytest.mark.parametrize("attribution", ["missing", "malformed", "foreign"])
@pytest.mark.parametrize("newer_lock", [False, True])
async def test_dispatch_invalid_attribution_closes_job_without_billing(
    fake_redis, fake_dream_db, attribution, newer_lock
):
    _, state_store, string_store = fake_redis
    now = datetime.now(timezone.utc)
    await persist_input_bundle(
        "p1",
        DreamInput(
            user_id="owner",
            expert_id="expert-original",
            group_id="expert_original",
            window_start=now,
            window_end=now,
        ),
        lock_token="original-token",
    )
    await write_initial_status(kind="dream_pass", job_id="j1", user_id="owner")
    _seed_submitted_pass(fake_dream_db)
    lock_key = MemoryScope.for_expert("owner", "expert-original").redis_key(
        "dream_lock"
    )
    string_store[lock_key] = "newer-token" if newer_lock else "original-token"
    string_store["dream:inflight:u1"] = "unrelated-token"
    state_store["dream:batch:state:p1"] = {
        "consolidate": json.dumps({"input_tokens": 10, "output_tokens": 20})
    }
    entry = _entry(phase="sanitize")
    if attribution == "missing":
        del entry.payload["cost_context"]
    elif attribution == "malformed":
        entry.payload["cost_context"] = {"generation": "paid-now"}
    else:
        entry.payload["cost_context"] = {"user_id": "foreign", "trial_id": None}

    with (
        patch("backend.copilot.dream.batch_costs.record_phase_cost") as bill,
        patch("backend.copilot.dream.batch_callbacks.submit_phase") as submit,
        patch("backend.copilot.dream.apply.apply_operations") as apply,
        patch("backend.copilot.trial_cost_context.capture_cost_context") as capture,
    ):
        await _dispatch(
            entry, [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)]
        )

    status = await read_status(kind="dream_pass", job_id="j1")
    assert status is not None and status.state == "errored"
    assert status.completed_at is not None
    assert string_store.get(lock_key) == ("newer-token" if newer_lock else None)
    assert string_store["dream:inflight:u1"] == "unrelated-token"
    row = fake_dream_db.rows["p1"]
    assert row["status"] is DreamPassStatus.ERRORED
    assert row["error"] == "batch: usage_attribution_unavailable"
    assert input_bundle_key("p1") not in string_store
    assert "dream:batch:state:p1" not in state_store
    bill.assert_not_called()
    submit.assert_not_called()
    apply.assert_not_called()
    capture.assert_not_called()


async def _a_pass_with_consolidate_landed(fake_redis) -> None:
    """Pass ``p1`` of u1 at sanitize: its bundle, its lock, its admin job and
    consolidate landed in its state."""
    _, state_store, string_store = fake_redis
    now = datetime.now(timezone.utc)
    await persist_input_bundle(
        "p1",
        DreamInput(user_id="u1", group_id="user_u1", window_start=now, window_end=now),
        lock_token="tok-u1",
    )
    await write_initial_status(kind="dream_pass", job_id="j1", user_id="u1")
    string_store[MemoryScope.for_user("u1").redis_key("dream_lock")] = "tok-u1"
    state_store["dream:batch:state:p1"] = {
        "consolidate": json.dumps({"input_tokens": 10, "output_tokens": 20})
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row_attribution",
    [None, {"user_id": "u1", "trial_id": "trial-other", "generation": None}],
    ids=["row_kept_none", "row_disagrees"],
)
async def test_a_result_its_row_does_not_attribute_ends_the_pass(
    fake_redis, fake_dream_db, row_attribution
):
    """The row is the source of the pass's attribution: a row that carries
    none (written before it kept one) or another than the batch's copy ends
    the pass unapplied. Its landed phase is charged only by the row: never
    when it carries none, and under the row's own when the copy differs."""
    await _a_pass_with_consolidate_landed(fake_redis)
    fake_dream_db.rows["p1"]["cost_context"] = row_attribution
    seen = []

    async def bill(ctx, usage):
        seen.append(get_trial_cost_context("u1"))

    with (
        patch("backend.copilot.dream.batch_costs.record_phase_cost", bill),
        patch(
            "backend.copilot.dream.cleanup.provider_batch_stopped",
            AsyncMock(return_value=True),
        ),
        patch("backend.copilot.dream.batch_callbacks.submit_phase") as submit,
        patch("backend.copilot.dream.apply.apply_operations") as apply,
    ):
        with pytest.raises(UsageActivationUnavailable):
            await handle_dream_batch_result(
                _entry(phase="sanitize"),
                [_row(custom_id="p1:sanitize", content=_SANITIZE_CONTENT)],
            )

    row = fake_dream_db.rows["p1"]
    assert row["status"] is DreamPassStatus.ERRORED
    assert row["error"] == "batch: usage_attribution_unavailable"
    assert [c.trial_id if c else None for c in seen] == (
        [] if row_attribution is None else ["trial-other"]
    )
    submit.assert_not_called()
    apply.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped", [False, True], ids=["running", "stopped"])
async def test_an_unattributed_result_stops_its_batch_before_the_mark_goes(
    fake_redis, fake_dream_db, stopped
):
    """The executor gave up waiting on a batch (its lifetime timeout) whose
    result carries no attribution: the pass ends, and its cleanup asks the
    provider to stop the batch the result came from. The row stays marked
    for the reaper until the provider says that batch has stopped."""
    await _a_pass_with_consolidate_landed(fake_redis)
    entry = _entry(phase="recombine")
    del entry.payload["cost_context"]
    stop = AsyncMock(return_value=stopped)
    charge = AsyncMock()

    with (
        patch("backend.copilot.dream.cleanup.provider_batch_stopped", stop),
        patch("backend.copilot.dream.batch_costs.record_phase_cost", charge),
    ):
        with pytest.raises(UsageActivationUnavailable):
            await handle_dream_batch_result(
                entry, [_row(custom_id="p1:recombine", content="", error="expired")]
            )

    stop.assert_awaited_once_with("msgbatch_recombine")
    charge.assert_awaited_once()
    assert charge.await_args.args[0].job.phase == "consolidate"
    row = fake_dream_db.rows["p1"]
    assert row["status"] is DreamPassStatus.ERRORED
    assert (row["cleanup_pending_at"] is None) is stopped
