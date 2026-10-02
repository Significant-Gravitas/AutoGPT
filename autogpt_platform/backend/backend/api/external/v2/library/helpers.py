"""
V2 External API - Library route helpers

Shared logic for the endpoints that start runs.
"""

from fastapi import HTTPException
from starlette import status

from backend.data.credit import get_credit_model

from ..tenancy import TenantContext


async def assert_can_pay(auth: TenantContext) -> None:
    """Refuse a run on a zero balance, as the internal run route does.

    Not the same guard as the paywall inside `add_graph_execution`, which asks
    whether the user has a subscription at all.
    """
    credit_model = await get_credit_model(auth.user_id, auth.organization_id)
    if await credit_model.get_credits(auth.user_id) <= 0:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient balance to execute the agent. "
            "Please top up your account.",
        )
