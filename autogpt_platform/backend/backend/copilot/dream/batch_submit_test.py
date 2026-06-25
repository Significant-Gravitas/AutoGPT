"""Tests for dream batch submission.

Covers the orphan-prevention guard (a paid provider batch must be
cancelled when the BatchExecutor enqueue fails afterwards — otherwise it
runs to completion with no callback to consume it) and the dream-lock
ownership token riding on the persisted input bundle so the batch
callback can compare-and-delete the lock hours later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.dream.batch_submit import (
    input_bundle_key,
    persist_input_bundle,
    read_input_bundle,
    read_lock_token,
    submit_phase,
)
from backend.copilot.dream.fetch import DreamInput
from backend.copilot.dream.locks import DREAM_LOCK_KEY_PREFIX
from backend.util.llm.providers import BatchSubmissionRef


def _bundle() -> DreamInput:
    now = datetime.now(timezone.utc)
    return DreamInput(
        user_id="u1", group_id="user_u1", window_start=now, window_end=now
    )


@pytest.fixture
def fake_redis():
    """Dict-backed redis stub matching the batch_callbacks_test pattern."""
    string_store: dict[str, str] = {}

    async def fake_get(key):
        return string_store.get(key)

    async def fake_set(key, value, ex=None, nx=False):
        if nx and key in string_store:
            return None
        string_store[key] = value
        return True

    async def fake_delete(key):
        string_store.pop(key, None)

    stub = AsyncMock()
    stub.get.side_effect = fake_get
    stub.set.side_effect = fake_set
    stub.delete.side_effect = fake_delete

    with patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(return_value=stub),
    ):
        yield stub, string_store


@pytest.mark.asyncio
async def test_enqueue_failure_cancels_orphaned_batch_and_reraises():
    submitted = BatchSubmissionRef(
        provider="anthropic",
        provider_batch_id="msgbatch_orphan",
        custom_id="p1_consolidate",
        submitted_at=datetime.now(timezone.utc),
    )
    call = AsyncMock(return_value=submitted)
    enqueue = AsyncMock(side_effect=RuntimeError("redis down"))
    cancel = AsyncMock(return_value=True)

    with patch("backend.copilot.dream.batch_submit.call_provider", call), patch(
        "backend.copilot.dream.batch_submit.enqueue_pending", enqueue
    ), patch("backend.copilot.dream.batch_submit.cancel_batch", cancel):
        with pytest.raises(RuntimeError, match="redis down"):
            await submit_phase(
                user_id="u1",
                pass_id="p1",
                job_id="j1",
                phase="consolidate",
                phase_models={"consolidate": "claude-sonnet-4-6"},
                api_key="sk-test",
                input_bundle=_bundle(),
            )

    # The just-submitted, paid batch must be cancelled, not left orphaned.
    cancel.assert_awaited_once()
    assert cancel.call_args.kwargs["provider_batch_id"] == "msgbatch_orphan"
    assert cancel.call_args.kwargs["api_key"] == "sk-test"


@pytest.mark.asyncio
async def test_successful_enqueue_does_not_cancel():
    submitted = BatchSubmissionRef(
        provider="anthropic",
        provider_batch_id="msgbatch_ok",
        custom_id="p1_consolidate",
        submitted_at=datetime.now(timezone.utc),
    )
    call = AsyncMock(return_value=submitted)
    enqueue = AsyncMock(return_value=None)
    cancel = AsyncMock()

    with patch("backend.copilot.dream.batch_submit.call_provider", call), patch(
        "backend.copilot.dream.batch_submit.enqueue_pending", enqueue
    ), patch("backend.copilot.dream.batch_submit.cancel_batch", cancel):
        ref = await submit_phase(
            user_id="u1",
            pass_id="p1",
            job_id="j1",
            phase="consolidate",
            phase_models={"consolidate": "claude-sonnet-4-6"},
            api_key="sk-test",
            input_bundle=_bundle(),
        )

    enqueue.assert_awaited_once()
    cancel.assert_not_awaited()
    assert ref.provider_batch_id == "msgbatch_ok"


@pytest.mark.asyncio
async def test_persist_input_bundle_carries_explicit_lock_token(fake_redis):
    """The orchestrator persists the bundle while still holding the dream
    lock and passes its OWN handle token; the bundle must carry it so the
    batch callback — hours later, in another process — can
    compare-and-delete."""
    _, string_store = fake_redis
    string_store[f"{DREAM_LOCK_KEY_PREFIX}u1"] = "tok-abc"

    await persist_input_bundle("p1", _bundle(), lock_token="tok-abc")

    assert await read_lock_token("p1") == "tok-abc"
    # The bundle itself still round-trips untouched by the extra field.
    bundle = await read_input_bundle("p1")
    assert bundle is not None
    assert bundle.user_id == "u1"


@pytest.mark.asyncio
async def test_persist_input_bundle_explicit_token_wins_over_live_key(fake_redis):
    """If this pass's lock expired and a NEWER pass re-acquired the key
    before persist runs, the live key holds the newer pass's token. The
    bundle must store the token the caller actually owns — storing the
    live value would let this pass's callback compare-and-delete the
    newer pass's lock hours later."""
    _, string_store = fake_redis
    string_store[f"{DREAM_LOCK_KEY_PREFIX}u1"] = "tok-newer-pass"

    await persist_input_bundle("p1", _bundle(), lock_token="tok-ours")

    assert await read_lock_token("p1") == "tok-ours"


@pytest.mark.asyncio
async def test_persist_input_bundle_falls_back_to_live_key_without_token(fake_redis):
    """Callers that don't supply a token (eval harness) fall back to
    reading the live lock key at persist time."""
    _, string_store = fake_redis
    string_store[f"{DREAM_LOCK_KEY_PREFIX}u1"] = "tok-live"

    await persist_input_bundle("p1", _bundle())

    assert await read_lock_token("p1") == "tok-live"


@pytest.mark.asyncio
async def test_persist_input_bundle_omits_token_when_lock_unheld(fake_redis):
    """No token supplied and no held lock at persist time ⇒ no token
    stored, and the callback falls back to the lock TTL instead of a
    blind delete."""
    await persist_input_bundle("p2", _bundle())

    assert await read_lock_token("p2") is None


@pytest.mark.asyncio
async def test_read_lock_token_none_when_bundle_expired(fake_redis):
    assert await read_lock_token("p-gone") is None


@pytest.mark.asyncio
async def test_read_lock_token_none_when_bundle_corrupted(fake_redis):
    _, string_store = fake_redis
    string_store[input_bundle_key("p3")] = "not json {{{"

    assert await read_lock_token("p3") is None
