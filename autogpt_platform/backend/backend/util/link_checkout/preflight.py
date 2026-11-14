"""Blockers Link would hit anyway, found before the customer is asked to
approve: a verification their Link account still needs, and the limits they
set in Link on what agents may spend (per purchase, per day, per 30 days).

Read from ``GET /userinfo``. Link enforces both regardless; this only saves the
customer an approval that cannot succeed and tells the agent why. Anything
missing or unreadable lets the purchase go ahead.
https://docs.stripe.com/agentic-commerce/link-agent-wallet/use-link-wallet-pay-online#next-actions
"""

import logging

import httpx
from pydantic import AliasChoices, BaseModel, Field, ValidationError

from backend.util.link_checkout.link import (
    LINK_API_BASE_URL,
    LINK_HTTP_TIMEOUT,
    link_action_url,
)

logger = logging.getLogger(__name__)

# Statuses that stop agent payments until the customer acts in Link.
_VERIFICATION_STEPS = {
    "ssn_verification": "verify their identity",
    "identity_verification": "verify their identity",
    "contact_support": "contact Link support",
}


class _Limit(BaseModel):
    # None means unlimited; finite values are in the smallest currency unit.
    limit: int | None = None
    remaining: int | None = None


class _SpendLimits(BaseModel):
    per_transaction: _Limit | None = None
    daily: _Limit | None = None
    thirty_day: _Limit | None = None


class _Verification(BaseModel):
    status: str = ""
    action_url: str | None = None


class _UserInfo(BaseModel):
    agent_wallet_spend_limits: _SpendLimits | None = None
    agent_wallet_verification_requirement: _Verification | None = Field(
        default=None,
        # Older responses name it `agent_wallet_step_up`.
        validation_alias=AliasChoices(
            "agent_wallet_verification_requirement", "agent_wallet_step_up"
        ),
    )


class PurchaseBlocker(BaseModel):
    message: str
    action_url: str = ""


async def purchase_blocker(
    access_token: str, amount: int, currency: str
) -> PurchaseBlocker | None:
    info = await _fetch_user_info(access_token)
    if info is None:
        return None
    verification = info.agent_wallet_verification_requirement
    if verification and verification.status in _VERIFICATION_STEPS:
        url = link_action_url(verification.action_url)
        step = _VERIFICATION_STEPS[verification.status]
        return PurchaseBlocker(
            message=f"Link needs the customer to {step} before an agent can pay. "
            + (f"Give them this link: {url} " if url else "Ask them to open Link. ")
            + "Prepare the purchase again once they are done.",
            action_url=url,
        )
    limits = info.agent_wallet_spend_limits
    # `/userinfo` gives limits without a currency; they are compared only for
    # US dollars, the currency Link reports them in for US accounts.
    if limits is None or currency != "usd":
        return None
    for name, limit, field in (
        ("per-purchase", limits.per_transaction, "limit"),
        ("remaining daily", limits.daily, "remaining"),
        ("remaining 30-day", limits.thirty_day, "remaining"),
    ):
        allowed = getattr(limit, field) if limit else None
        if allowed is not None and amount > allowed:
            return PurchaseBlocker(
                message=f"This purchase ({_dollars(amount)}) is over the "
                f"customer's {name} limit for agent spending in Link "
                f"({_dollars(allowed)}). Ask them to raise it in the Link app, "
                "or choose a smaller purchase."
            )
    return None


def _dollars(cents: int) -> str:
    return f"${cents / 100:,.2f}"


async def _fetch_user_info(access_token: str) -> _UserInfo | None:
    try:
        async with httpx.AsyncClient(
            timeout=LINK_HTTP_TIMEOUT, trust_env=False, follow_redirects=False
        ) as client:
            response = await client.get(
                f"{LINK_API_BASE_URL}/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        if response.status_code != 200:
            logger.info(f"Link userinfo unavailable: HTTP {response.status_code}")
            return None
        return _UserInfo.model_validate_json(response.content)
    except (httpx.HTTPError, ValidationError) as e:
        logger.info(f"Link userinfo unreadable: {type(e).__name__}")
        return None
