"""Platform info tool — on-demand subscription and billing data for AutoPilot."""

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.copilot.rate_limit import (
    SubscriptionTier,
    get_tier_multipliers,
    get_user_tier,
    get_workspace_storage_limits_mb,
)

from .base import BaseTool
from .models import ErrorResponse, PlatformInfoResponse, ToolResponseBase

logger = logging.getLogger(__name__)

# Human-friendly tier descriptions for upgrade suggestions.
_TIER_DESCRIPTIONS: dict[SubscriptionTier, str] = {
    SubscriptionTier.NO_TIER: "No active subscription",
    SubscriptionTier.BASIC: "Basic",
    SubscriptionTier.PRO: "Pro — 5× usage limits",
    SubscriptionTier.MAX: "Max — 20× usage limits, 5 GB storage",
    SubscriptionTier.BUSINESS: "Business — 60× usage limits, 15 GB storage",
    SubscriptionTier.ENTERPRISE: "Enterprise — 60× usage limits, 15 GB storage",
}

# Ordered list for upgrade suggestions (show tiers above current).
_TIER_ORDER: list[SubscriptionTier] = [
    SubscriptionTier.NO_TIER,
    SubscriptionTier.BASIC,
    SubscriptionTier.PRO,
    SubscriptionTier.MAX,
    SubscriptionTier.BUSINESS,
    SubscriptionTier.ENTERPRISE,
]


class PlatformInfoTool(BaseTool):
    """Provides subscription tier, limits, and billing info on demand."""

    @property
    def name(self) -> str:
        return "get_platform_info"

    @property
    def description(self) -> str:
        return (
            "Get the user's subscription plan, usage limits, and billing info. "
            "Call when the user asks about their plan, limits, billing, or upgrading."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": ["subscription"],
                    "description": "The platform topic to query.",
                },
            },
            "required": ["topic"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        topic: str = "subscription",
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id

        if topic == "subscription":
            return await self._handle_subscription(user_id, session_id)

        return ErrorResponse(
            message=f"Unknown topic '{topic}'. Available topics: subscription",
            error="invalid_topic",
            session_id=session_id,
        )

    async def _handle_subscription(
        self,
        user_id: str | None,
        session_id: str | None,
    ) -> ToolResponseBase:
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="missing_user_id",
                session_id=session_id,
            )

        try:
            tier = await get_user_tier(user_id)
        except Exception:
            logger.exception("Failed to fetch user tier for user %s", user_id)
            return ErrorResponse(
                message="Could not retrieve subscription info.",
                error="tier_lookup_failed",
                session_id=session_id,
            )

        tier_multipliers = await get_tier_multipliers()
        storage_limits = await get_workspace_storage_limits_mb()
        multiplier = tier_multipliers.get(tier.value, 1.0)
        storage_mb = storage_limits.get(tier.value, 250)

        # Build upgrade suggestions: show tiers above the current one.
        current_idx = _TIER_ORDER.index(tier) if tier in _TIER_ORDER else 0
        upgrade_options = [
            {"tier": t.value, "description": _TIER_DESCRIPTIONS[t]}
            for t in _TIER_ORDER[current_idx + 1 :]
            if t != SubscriptionTier.ENTERPRISE  # not self-serve
        ]

        return PlatformInfoResponse(
            message=f"Your current plan: {tier.value}",
            topic="subscription",
            tier=tier.value,
            tier_multiplier=multiplier,
            workspace_storage_mb=storage_mb,
            billing_url="/settings/billing",
            data={
                "upgrade_options": upgrade_options,
                "manage_billing_url": "/settings/billing",
            },
            session_id=session_id,
        )
