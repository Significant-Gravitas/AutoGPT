from unittest.mock import AsyncMock

import pytest

from . import work_items


@pytest.mark.asyncio
async def test_terminal_report_always_claims_one_parent_wake(monkeypatch) -> None:
    claim = AsyncMock(side_effect=[True, False])
    monkeypatch.setattr(work_items, "claim_parent_wake", claim)

    first = await work_items.should_enqueue_parent_wake("work-1", "user-1")
    duplicate = await work_items.should_enqueue_parent_wake("work-1", "user-1")

    assert first is True
    assert duplicate is False
    assert claim.await_count == 2
