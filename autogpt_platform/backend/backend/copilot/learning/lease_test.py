from unittest.mock import AsyncMock

import pytest

from backend.copilot.learning import nightly


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner, unavailable",
    [
        ("another-worker", False),
        (None, False),
        (None, True),
    ],
)
async def test_only_confirmed_lease_contention_is_reported_as_held(
    monkeypatch, owner, unavailable
):
    redis = AsyncMock()
    redis.set.return_value = False
    redis.get.return_value = owner
    if unavailable:
        redis.set.side_effect = ConnectionError("Redis unavailable")
    monkeypatch.setattr(nightly, "get_redis_async", AsyncMock(return_value=redis))
    monkeypatch.setattr(nightly, "is_feature_enabled", AsyncMock(return_value=True))
    run = AsyncMock()
    monkeypatch.setattr(nightly, "_run_under_lease", run)
    result = await nightly.run_skill_learning_pass("lease-test-user")
    run.assert_not_awaited()
    if owner:
        assert result.skip_reason == "lease_held" and not result.error
    else:
        assert not result.skipped and result.skip_reason != "lease_held"
        assert result.error == "RuntimeError: skill learning lease unavailable"
