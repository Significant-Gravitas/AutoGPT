from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from prisma.enums import SubscriptionTier

from backend.copilot.rate_limit import UserPaywalledError
from backend.integrations.codex import access
from backend.util.entitlements import Entitlement, EntitlementRequiredError


@pytest.mark.asyncio
async def test_access_predicate_uses_codex_transport_entitlement():
    with patch.object(
        access,
        "has_entitlement",
        new_callable=AsyncMock,
        return_value=True,
    ) as has_entitlement:
        assert await access.has_codex_access("user-1")

    has_entitlement.assert_awaited_once_with(
        "user-1",
        Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
    )


@pytest.mark.asyncio
async def test_execution_gate_translates_entitlement_error_to_paywall_error():
    denied = EntitlementRequiredError(
        Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        SubscriptionTier.MAX,
    )
    with patch.object(
        access,
        "require_entitlement",
        new_callable=AsyncMock,
        side_effect=denied,
    ):
        with pytest.raises(UserPaywalledError, match="Max plan"):
            await access.enforce_codex_access("user-1")


@pytest.mark.asyncio
async def test_http_gate_translates_entitlement_error_to_payment_required():
    denied = EntitlementRequiredError(
        Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        SubscriptionTier.MAX,
    )
    with patch.object(
        access,
        "require_entitlement",
        new_callable=AsyncMock,
        side_effect=denied,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await access.enforce_codex_access_http("user-1")

    assert exc_info.value.status_code == 402
    assert exc_info.value.detail == access.CODEX_MINIMUM_PLAN_ERROR


@pytest.mark.asyncio
async def test_access_gate_propagates_tier_lookup_errors():
    with patch.object(
        access,
        "require_entitlement",
        new_callable=AsyncMock,
        side_effect=RuntimeError("database unavailable"),
    ):
        with pytest.raises(RuntimeError, match="database unavailable"):
            await access.enforce_codex_access("user-1")


@pytest.mark.asyncio
async def test_http_gate_maps_tier_lookup_errors_to_service_unavailable():
    with patch.object(
        access,
        "require_entitlement",
        new_callable=AsyncMock,
        side_effect=RuntimeError("database unavailable"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await access.enforce_codex_access_http("user-1")

    assert exc_info.value.status_code == 503
    assert exc_info.value.headers == {"Retry-After": "30"}


@pytest.mark.asyncio
async def test_discovery_predicate_hides_codex_on_tier_lookup_errors():
    with patch.object(
        access,
        "has_codex_access",
        new_callable=AsyncMock,
        side_effect=RuntimeError("database unavailable"),
    ):
        allowed = await access.has_codex_access_for_discovery("user-1")

    assert allowed is False
