"""Whether someone may use paid features, and at which tier: the one check.

Every route, background job and Otto tool asks here, passing the
organization the work runs in:

- **A personal organization, or no organization known**: the user's own plan
  (``subscriptionTier``, trials included), exactly as
  ``backend.copilot.rate_limit`` decides it. With ``ENABLE_PLATFORM_PAYMENT``
  off for the user, nobody is paywalled.
- **A team organization**: the organization's plan (``backend.data.org_plan``)
  and an active membership. The person's own plan doesn't count there, so an
  owner with personal Pro is paywalled in their team organization until it has
  a plan, and the refusal asks for the organization's plan, never for a
  personal upgrade. With payments off, members aren't paywalled either.

Otto's usage limits and workspace storage follow the tier decided here, per
person: each seat gets the whole tier's limits (``entitlement_limits.py``).
Lookup errors propagate: HTTP gates answer 503 with ``Retry-After`` and
background work retries. ``entitlement_architecture_test.py`` refuses calls to
the per-user checks from anywhere else.
"""

import logging
from typing import Literal

import fastapi
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.models import RequestContext
from pydantic import BaseModel

from backend.copilot import rate_limit
from backend.copilot.rate_limit import SubscriptionTier
from backend.data.db_accessors import org_plan_db
from backend.data.org_plan import OrgPlanSnapshot, get_org_plan_snapshot
from backend.util.exceptions import UserPaywalledError
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

PaywallReason = Literal["no_plan", "org_plan_required", "not_member", "org_inactive"]

_REFUSALS: dict[PaywallReason, str] = {
    "no_plan": "A subscription is required to {action}. Upgrade to continue.",
    "org_plan_required": (
        "This organization needs an active plan to {action}. Ask its owner to "
        "contact AutoGPT about a team plan."
    ),
    "not_member": "You aren't an active member of this organization.",
    "org_inactive": "This organization is no longer active.",
}


class Entitlement(BaseModel):
    user_id: str
    org_id: str | None
    # Whose plan decides: the user's own, or the team organization's.
    scope: Literal["personal", "organization"]
    tier: SubscriptionTier
    paywall: PaywallReason | None = None

    @property
    def paywalled(self) -> bool:
        return self.paywall is not None


def paywall_message(reason: PaywallReason, action: str = "use this feature") -> str:
    """What a refused person reads, in the 402 or the background refusal."""
    return _REFUSALS[reason].format(action=action)


async def get_entitlement(user_id: str, org_id: str | None = None) -> Entitlement:
    """Who decides, whether *user_id* is refused, and the tier they get."""
    team = await team_org_snapshot(org_id)
    if team is None:
        return Entitlement(
            user_id=user_id,
            org_id=org_id,
            scope="personal",
            paywall="no_plan" if await rate_limit.is_user_paywalled(user_id) else None,
            tier=await rate_limit.get_user_tier(user_id),
        )
    reason = await _org_paywall_reason(user_id, team)
    return Entitlement(
        user_id=user_id,
        org_id=org_id,
        scope="organization",
        tier=_plan_tier(team) if reason is None else SubscriptionTier.NO_TIER,
        paywall=reason,
    )


async def paywall_reason(
    user_id: str, org_id: str | None = None
) -> PaywallReason | None:
    """Why *user_id* is refused paid features in *org_id*, or None."""
    team = await team_org_snapshot(org_id)
    if team is None:
        return "no_plan" if await rate_limit.is_user_paywalled(user_id) else None
    return await _org_paywall_reason(user_id, team)


async def is_user_paywalled(user_id: str, org_id: str | None = None) -> bool:
    """Whether *user_id* is refused paid features for work in *org_id*."""
    return await paywall_reason(user_id, org_id) is not None


async def get_user_tier(user_id: str, org_id: str | None = None) -> SubscriptionTier:
    """The tier *user_id* works under in *org_id*: their own plan's in a
    personal organization, the plan's in a team organization."""
    team = await team_org_snapshot(org_id)
    if team is None:
        return await rate_limit.get_user_tier(user_id)
    if await _org_paywall_reason(user_id, team) is not None:
        return SubscriptionTier.NO_TIER
    return _plan_tier(team)


async def require_entitlement(
    user_id: str, org_id: str | None = None, *, action: str = "use this feature"
) -> None:
    """Background gate: raise ``UserPaywalledError`` when refused. Lookup
    errors propagate, so the caller's retry decides."""
    reason = await paywall_reason(user_id, org_id)
    if reason is not None:
        raise UserPaywalledError(paywall_message(reason, action))


async def enforce_entitlement(
    user_id: str, org_id: str | None = None, *, action: str = "use this feature"
) -> None:
    """HTTP gate: 402 when refused, 503 with ``Retry-After`` when the plan
    can't be read."""
    try:
        reason = await paywall_reason(user_id, org_id)
    except Exception as exc:
        logger.warning(
            f"Entitlement lookup failed for user {user_id[:8]} in org {org_id}: {exc}"
        )
        raise fastapi.HTTPException(
            status_code=503,
            detail="Subscription state temporarily unavailable, retry shortly.",
            headers={"Retry-After": "30"},
        ) from exc
    if reason is not None:
        raise UserPaywalledError(paywall_message(reason, action))


async def enforce_payment_paywall(
    ctx: RequestContext = fastapi.Security(get_request_context),
) -> None:
    """Route dependency: ``dependencies=[Depends(enforce_payment_paywall)]``.

    Reads the verified request context, so the organization the request acts
    in decides. Routes authenticated another way (API keys, the chat routes
    whose model route decides) call :func:`enforce_entitlement` with the
    organization they act in.
    """
    await enforce_entitlement(ctx.user_id, ctx.org_id)


async def team_org_snapshot(org_id: str | None) -> OrgPlanSnapshot | None:
    """The organization's snapshot when its plan decides, else None (no
    organization, or a personal one)."""
    if not org_id:
        return None
    snapshot = await get_org_plan_snapshot(org_id)
    if snapshot.exists and snapshot.is_personal:
        return None
    return snapshot


async def _org_paywall_reason(
    user_id: str, team: OrgPlanSnapshot
) -> PaywallReason | None:
    if not team.exists or not team.is_live:
        return "org_inactive"
    if not await org_plan_db().query_active_membership(team.org_id, user_id):
        return "not_member"
    if team.counting_plan() is not None:
        return None
    if not await is_feature_enabled(Flag.ENABLE_PLATFORM_PAYMENT, user_id):
        return None
    return "org_plan_required"


def _plan_tier(team: OrgPlanSnapshot) -> SubscriptionTier:
    """The plan's tier, or NO_TIER (BASIC limits with payments off)."""
    plan = team.counting_plan()
    return plan.tier if plan is not None else SubscriptionTier.NO_TIER
