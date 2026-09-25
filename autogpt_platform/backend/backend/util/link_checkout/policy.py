"""Whether a purchase may be approved inside AutoGPT instead of in Link.

The customer sets an approval policy for AutoGPT in their Link wallet: per
purchase limits, optionally restricted to some payment methods. Within it,
AutoGPT can record the customer's approval itself and create the spend request
already approved (``POST /spend_requests/create_delegated``). Outside it, or if
the policy cannot be read, the customer approves in Link as before.
"""

import logging

import httpx
from pydantic import BaseModel, ValidationError

from backend.util.link_checkout.link import LINK_API_BASE_URL, LINK_HTTP_TIMEOUT
from backend.util.link_checkout.models import CheckoutPlan

logger = logging.getLogger(__name__)


class PurchaseLimit(BaseModel):
    amount: int
    currency: str


class RuleLimits(BaseModel):
    per_purchase: PurchaseLimit


class ApprovalRule(BaseModel):
    action: str
    limits: RuleLimits
    allowed_payment_methods: list[str] | None = None


class ApprovalPolicy(BaseModel):
    rules: list[ApprovalRule]


async def in_app_approval_allowed(access_token: str, plan: CheckoutPlan) -> bool:
    policy = await _fetch_policy(access_token)
    return policy is not None and any(_covers(rule, plan) for rule in policy.rules)


def _covers(rule: ApprovalRule, plan: CheckoutPlan) -> bool:
    limit = rule.limits.per_purchase
    return (
        rule.action == "spend_request_create"
        and limit.currency.lower() == plan.currency
        and plan.amount <= limit.amount
        and (
            not rule.allowed_payment_methods
            or plan.payment_method_id in rule.allowed_payment_methods
        )
    )


async def _fetch_policy(access_token: str) -> ApprovalPolicy | None:
    # Any failure falls back to approval in Link; nothing is lost but a click.
    try:
        async with httpx.AsyncClient(timeout=LINK_HTTP_TIMEOUT) as client:
            response = await client.get(
                f"{LINK_API_BASE_URL}/approval-policy",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        if response.status_code != 200:
            logger.info(
                "Link approval policy unavailable: HTTP %s", response.status_code
            )
            return None
        return ApprovalPolicy.model_validate_json(response.content)
    except (httpx.HTTPError, ValidationError) as e:
        logger.info("Link approval policy unreadable: %s", type(e).__name__)
        return None
