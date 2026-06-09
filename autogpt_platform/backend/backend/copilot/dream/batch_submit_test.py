"""Tests for dream batch submission.

Focused on the orphan-prevention guard: if the provider accepts a batch
submission but the BatchExecutor enqueue fails afterwards, the paid
provider batch must be cancelled before the error propagates — otherwise
it runs to completion with no callback to consume it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.dream.batch_submit import submit_phase
from backend.copilot.dream.fetch import DreamInput
from backend.util.llm.providers import BatchSubmissionRef


def _bundle() -> DreamInput:
    now = datetime.now(timezone.utc)
    return DreamInput(
        user_id="u1", group_id="user_u1", window_start=now, window_end=now
    )


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
