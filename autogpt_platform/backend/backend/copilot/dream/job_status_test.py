"""Tests for ``copilot.dream.job_status``."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.dream.job_status import (
    STATUS_KEY_PREFIX,
    STATUS_TTL_SECONDS,
    JobStatus,
    mark_complete,
    mark_errored,
    read_status,
    update_status_phase,
    write_initial_status,
)


@pytest.fixture
def fake_redis():
    """In-memory dict-backed redis stub for the registry tests."""
    store: dict[str, str] = {}

    async def fake_set(key, value, ex=None):
        store[key] = value
        return True

    async def fake_get(key):
        return store.get(key)

    redis_stub = AsyncMock()
    redis_stub.set.side_effect = fake_set
    redis_stub.get.side_effect = fake_get

    async def fake_get_redis_async():
        return redis_stub

    with patch(
        "backend.data.redis_client.get_redis_async",
        side_effect=fake_get_redis_async,
    ):
        yield redis_stub, store


class TestWriteInitialStatus:
    @pytest.mark.asyncio
    async def test_returns_status_in_queued_state(self, fake_redis):
        status = await write_initial_status(
            kind="nightly", job_id="job-1", user_id="user-x"
        )
        assert status.state == "queued"
        assert status.job_id == "job-1"
        assert status.user_id == "user-x"
        assert status.kind == "nightly"
        assert status.started_at == status.updated_at
        assert status.completed_at is None

    @pytest.mark.asyncio
    async def test_persists_with_ttl(self, fake_redis):
        redis, store = fake_redis
        await write_initial_status(kind="dream_pass", job_id="job-2", user_id="user-y")
        # Single SET with TTL — no MULTI/HSET dance
        assert redis.set.await_count == 1
        kwargs = redis.set.await_args.kwargs
        assert kwargs["ex"] == STATUS_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_uses_namespaced_key(self, fake_redis):
        _, store = fake_redis
        await write_initial_status(kind="rebuild", job_id="job-3", user_id="user-z")
        assert f"{STATUS_KEY_PREFIX}:rebuild:job-3" in store


class TestUpdateStatusPhase:
    @pytest.mark.asyncio
    async def test_updates_state_and_current_phase(self, fake_redis):
        await write_initial_status(kind="dream_pass", job_id="j1", user_id="u")
        updated = await update_status_phase(
            kind="dream_pass",
            job_id="j1",
            state="running",
            current_phase="consolidate",
        )
        assert updated is not None
        assert updated.state == "running"
        assert updated.current_phase == "consolidate"

    @pytest.mark.asyncio
    async def test_partial_update_preserves_other_fields(self, fake_redis):
        """Updating just current_phase shouldn't reset state, batch_id, etc."""
        await write_initial_status(kind="dream_pass", job_id="j1", user_id="u")
        await update_status_phase(kind="dream_pass", job_id="j1", state="running")
        await update_status_phase(
            kind="dream_pass", job_id="j1", current_phase="recombine"
        )
        result = await read_status(kind="dream_pass", job_id="j1")
        assert result is not None
        assert result.state == "running"  # not overwritten by second update
        assert result.current_phase == "recombine"

    @pytest.mark.asyncio
    async def test_advances_updated_at(self, fake_redis):
        await write_initial_status(kind="dream_pass", job_id="j1", user_id="u")
        initial = await read_status(kind="dream_pass", job_id="j1")
        assert initial is not None
        # Wait a tick — datetime.now resolution on macOS is fine-grained enough
        import asyncio

        await asyncio.sleep(0.001)
        updated = await update_status_phase(
            kind="dream_pass", job_id="j1", current_phase="sanitize"
        )
        assert updated is not None
        assert updated.updated_at > initial.updated_at

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_row(self, fake_redis):
        """A missing row is a soft failure — return None, don't raise.
        Work bodies must not crash because a TTL expired mid-run."""
        result = await update_status_phase(
            kind="dream_pass", job_id="never-existed", state="running"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_sets_batch_id_when_provided(self, fake_redis):
        await write_initial_status(kind="dream_pass", job_id="j1", user_id="u")
        updated = await update_status_phase(
            kind="dream_pass",
            job_id="j1",
            state="submitted",
            batch_id="msgbatch_abc123",
        )
        assert updated is not None
        assert updated.batch_id == "msgbatch_abc123"
        assert updated.state == "submitted"


class TestMarkComplete:
    @pytest.mark.asyncio
    async def test_transitions_to_complete_with_result(self, fake_redis):
        await write_initial_status(kind="nightly", job_id="j1", user_id="u")
        await update_status_phase(kind="nightly", job_id="j1", state="running")
        updated = await mark_complete(
            kind="nightly",
            job_id="j1",
            result={"dream": "ran", "ratification": "ran"},
        )
        assert updated is not None
        assert updated.state == "complete"
        assert updated.completed_at is not None
        assert updated.result == {"dream": "ran", "ratification": "ran"}

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_row(self, fake_redis):
        result = await mark_complete(kind="nightly", job_id="ghost", result={})
        assert result is None

    @pytest.mark.asyncio
    async def test_refuses_to_overwrite_errored_job(self, fake_redis):
        """Terminal→terminal is rejected: a late mark_complete must not
        rewrite a job that already failed (symmetric with mark_errored's
        complete guard)."""
        await write_initial_status(kind="dream_pass", job_id="j1", user_id="u")
        await mark_errored(kind="dream_pass", job_id="j1", error="boom")
        result = await mark_complete(kind="dream_pass", job_id="j1", result={"n": 1})
        assert result is not None
        assert result.state == "errored"
        assert result.error == "boom"


class TestMarkErrored:
    @pytest.mark.asyncio
    async def test_transitions_to_errored_with_short_message(self, fake_redis):
        await write_initial_status(kind="dream_pass", job_id="j1", user_id="u")
        updated = await mark_errored(
            kind="dream_pass", job_id="j1", error="LLM returned non-JSON"
        )
        assert updated is not None
        assert updated.state == "errored"
        assert updated.error == "LLM returned non-JSON"

    @pytest.mark.asyncio
    async def test_caps_error_string_at_2000_chars(self, fake_redis):
        """Stack traces can be huge; cap to keep Redis values reasonable."""
        await write_initial_status(kind="dream_pass", job_id="j1", user_id="u")
        huge_error = "x" * 5000
        updated = await mark_errored(kind="dream_pass", job_id="j1", error=huge_error)
        assert updated is not None
        assert updated.error is not None
        assert len(updated.error) == 2000

    @pytest.mark.asyncio
    async def test_refuses_to_overwrite_completed_job(self, fake_redis):
        """A transient error routed through a crash guard after the success
        tail must not rewrite a completed job (whose writes landed) as
        errored."""
        await write_initial_status(kind="dream_pass", job_id="j1", user_id="u")
        await mark_complete(kind="dream_pass", job_id="j1", result={"n": 1})
        result = await mark_errored(kind="dream_pass", job_id="j1", error="late blip")
        assert result is not None
        assert result.state == "complete"
        assert result.result == {"n": 1}


class TestReadStatus:
    @pytest.mark.asyncio
    async def test_returns_none_for_missing_key(self, fake_redis):
        result = await read_status(kind="nightly", job_id="never-existed")
        assert result is None

    @pytest.mark.asyncio
    async def test_roundtrips_through_redis(self, fake_redis):
        """Pin the JSON serialization shape so a future Pydantic field
        rename doesn't silently corrupt existing rows."""
        await write_initial_status(kind="rebuild", job_id="j1", user_id="user-rt")
        result = await read_status(kind="rebuild", job_id="j1")
        assert result is not None
        assert result.user_id == "user-rt"
        assert result.kind == "rebuild"

    @pytest.mark.asyncio
    async def test_returns_none_for_corrupted_row(self, fake_redis):
        """A corrupted row (e.g., missing required fields after a schema
        change) should be treated as missing rather than crashing the
        admin GET endpoint."""
        redis, store = fake_redis
        store[f"{STATUS_KEY_PREFIX}:nightly:j1"] = (
            '{"this is": "not a valid JobStatus"}'
        )
        result = await read_status(kind="nightly", job_id="j1")
        assert result is None


class TestJobStatusModel:
    def test_default_terminal_fields_are_none(self):
        now = datetime.now(timezone.utc)
        s = JobStatus(
            job_id="x",
            user_id="u",
            kind="nightly",
            state="queued",
            started_at=now,
            updated_at=now,
        )
        assert s.completed_at is None
        assert s.result is None
        assert s.error is None
        assert s.batch_id is None
        assert s.current_phase is None
