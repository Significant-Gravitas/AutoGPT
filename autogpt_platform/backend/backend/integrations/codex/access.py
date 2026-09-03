import logging

from fastapi import HTTPException, status

from backend.copilot.rate_limit import UserPaywalledError
from backend.util.entitlements import (
    Entitlement,
    EntitlementRequiredError,
    has_entitlement,
    require_entitlement,
)

CODEX_MINIMUM_PLAN_ERROR = "A Max plan or higher is required to use ChatGPT."
logger = logging.getLogger(__name__)


async def has_codex_access(user_id: str) -> bool:
    return await has_entitlement(user_id, Entitlement.CODEX_SUBSCRIPTION_TRANSPORT)


async def has_codex_access_for_discovery(user_id: str) -> bool:
    try:
        return await has_codex_access(user_id)
    except Exception:
        logger.warning(
            "Unable to resolve Codex entitlement for user %s; hiding transport",
            user_id,
            exc_info=True,
        )
        return False


async def enforce_codex_access(user_id: str) -> None:
    try:
        await require_entitlement(
            user_id,
            Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        )
    except EntitlementRequiredError:
        raise UserPaywalledError(CODEX_MINIMUM_PLAN_ERROR) from None


async def enforce_codex_access_http(user_id: str) -> None:
    try:
        await require_entitlement(
            user_id,
            Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        )
    except EntitlementRequiredError:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=CODEX_MINIMUM_PLAN_ERROR,
        ) from None
    except Exception as exc:
        raise _subscription_state_unavailable() from exc


def _subscription_state_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Subscription state temporarily unavailable, retry shortly.",
        headers={"Retry-After": "30"},
    )
