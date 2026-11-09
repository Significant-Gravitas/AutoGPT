"""The per-user claim that records an Exa Agent run's cost only once."""

from unittest.mock import patch

import pytest

from backend.blocks.exa import agent_billing


@pytest.mark.asyncio
async def test_billing_claim_records_when_redis_is_unreachable():
    async def redis_down():
        raise ConnectionError("redis down")

    with patch.object(agent_billing, "get_redis_async", redis_down):
        assert await agent_billing.claim_run_billing("user-1", "agent_run_1")


@pytest.mark.asyncio
async def test_billing_claim_is_once_per_user_and_run(billing_claims):
    assert await agent_billing.claim_run_billing("user-1", "agent_run_1")
    assert not await agent_billing.claim_run_billing("user-1", "agent_run_1")
    assert await agent_billing.claim_run_billing("user-2", "agent_run_1")
