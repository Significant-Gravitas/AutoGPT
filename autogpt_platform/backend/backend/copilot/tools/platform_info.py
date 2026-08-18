"""Platform info tool — on-demand subscription and billing data for AutoPilot."""

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.copilot.rate_limit import get_user_tier

from .base import BaseTool
from .models import ErrorResponse, PlatformInfoResponse, ToolResponseBase

logger = logging.getLogger(__name__)

# Human-friendly tier names shown to the user.
_TIER_DISPLAY_NAMES: dict[str, str] = {
    "NO_TIER": "No active subscription",
    "BASIC": "Basic",
    "PRO": "Pro",
    "MAX": "Max",
    "BUSINESS": "Business",
    "ENTERPRISE": "Enterprise",
}


class PlatformInfoTool(BaseTool):
    """Provides the user's subscription tier and billing link on demand."""

    @property
    def name(self) -> str:
        return "get_platform_info"

    @property
    def description(self) -> str:
        return "Call when the user asks about their plan, billing, or account."

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

        # When billing is disabled (self-hosted, beta cohort), skip tier
        # lookup and report open access — there is no billing page to manage.
        from backend.util.feature_flag import Flag, is_feature_enabled

        billing_enabled = await is_feature_enabled(
            Flag.ENABLE_PLATFORM_PAYMENT, user_id
        )
        if not billing_enabled:
            return PlatformInfoResponse(
                message=(
                    "Billing is not enabled on this deployment. "
                    "You have open access to the platform. "
                    "You are AutoPilot, running on the AutoGPT platform. "
                    "Under the hood you may use models via OpenRouter or Claude SDK, "
                    "but only mention the AutoGPT platform to the user — "
                    "that is the only thing they can manage or change. "
                    "Never direct users to external AI provider billing pages."
                ),
                topic="subscription",
                tier="OPEN_ACCESS",
                billing_url=None,
                session_id=session_id,
            )

        try:
            tier = await get_user_tier(user_id)
        except Exception:
            logger.exception("Failed to fetch subscription info for user %s", user_id)
            return ErrorResponse(
                message="Could not retrieve subscription info.",
                error="tier_lookup_failed",
                session_id=session_id,
            )

        display_name = _TIER_DISPLAY_NAMES.get(tier.value, tier.value)

        return PlatformInfoResponse(
            message=(
                f"You are on the {display_name} plan. "
                "You can manage your billing and subscription at Settings → Billing. "
                "You are AutoPilot, running on the AutoGPT platform. "
                "Under the hood you may use models via OpenRouter or Claude SDK, "
                "but only mention the AutoGPT platform to the user — "
                "that is the only thing they can manage or change. "
                "Never direct users to external AI provider billing pages."
            ),
            topic="subscription",
            tier=tier.value,
            billing_url="/settings/billing",
            session_id=session_id,
        )
