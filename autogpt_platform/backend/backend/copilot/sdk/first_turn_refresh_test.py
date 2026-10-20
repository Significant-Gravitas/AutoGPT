import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti import context

from .retry_scenarios_test import TestFirstTurnWarmContextIsEphemeral as _FirstTurn
from .retry_scenarios_test import TestFollowUpWarmContextCallSite as _FollowUp

_SVC = "backend.copilot.sdk.service"
_FRESH = "<temporal_context>fresh retry memory</temporal_context>"


@pytest.mark.asyncio
@pytest.mark.parametrize("times_out", [False, True], ids=["empty", "timeout"])
@pytest.mark.parametrize("retry", [False, True], ids=["first-send", "overflow-retry"])
async def test_initial_empty_recall_is_final_until_a_retry(
    monkeypatch, times_out, retry
) -> None:
    monkeypatch.setattr(context.graphiti_config, "context_timeout", 0.02)

    async def fetch(*args, **kwargs):
        if times_out:
            await asyncio.Event().wait()
        return None

    fetch_mock = AsyncMock(side_effect=fetch)
    monkeypatch.setattr(context, "_fetch", fetch_mock)
    refresh = AsyncMock(return_value=_FRESH)
    queries: list[str] = []
    await _FirstTurn()._run(
        _FirstTurn._first_turn("hi"),
        _FollowUp._clients(queries, fail_first=retry),
        first_block=None,
        refresh=refresh,
        extra=[
            (f"{_SVC}.fetch_warm_context", dict(new=context.fetch_warm_context)),
            (f"{_SVC}.persist_and_record_usage", dict(new=AsyncMock())),
        ],
    )

    fetch_mock.assert_awaited_once()
    assert len(queries) == (2 if retry else 1)
    assert "temporal_context" not in queries[0]
    if retry:
        refresh.assert_awaited_once()
        args, kwargs = refresh.await_args
        assert args[0] == "test-user"
        assert args[1] == "hi"
        assert kwargs == {"expert_id": None, "force": True}
        assert "fresh retry memory" in queries[1]
    else:
        refresh.assert_not_awaited()
