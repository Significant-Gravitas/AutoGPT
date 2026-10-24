"""Say plainly that a service cannot be connected yet, and record the request.

Without this, a search that found nothing ended in prose ("nothing built in,
so it connects through its API") and a hand-built HTTP request asking a new
user for a raw token. Every request for the same service lands on one
feature request, one customer need per ask, so the gap is counted.
"""

import logging
from typing import Any

from backend.api.features.integrations.support import find_connectable
from backend.copilot.model import ChatSession

from .base import BaseTool
from .feature_requests import file_feature_request, find_feature_request
from .models import (
    ErrorResponse,
    IntegrationRequestResponse,
    ResponseType,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

TITLE_PREFIX = "Integration request: "
MAX_SERVICE_CHARS = 60
MAX_USE_CASE_CHARS = 500


class RequestIntegrationTool(BaseTool):
    @property
    def name(self) -> str:
        return "request_integration"

    @property
    def description(self) -> str:
        return (
            "Not-supported card for a service nothing covers; records the "
            "request. Call before any workaround."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "e.g. 'Shopify'"},
                "use_case": {
                    "type": "string",
                    "description": "What for, briefly. No personal info.",
                },
            },
            "required": ["service"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        service: str = "",
        use_case: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        name = " ".join((service or "").split())[:MAX_SERVICE_CHARS]
        if not name:
            return ErrorResponse(
                message="Name the service the user wants.", session_id=session_id
            )
        if not user_id:
            return ErrorResponse(
                message="Authentication required.", session_id=session_id
            )
        supported = find_connectable(name)
        if supported is not None:
            return ErrorResponse(
                message=(
                    f"{supported.name} can be connected: "
                    f"find_capability(query='{supported.key}') lists what it does. "
                    "Do not tell the user it is unsupported."
                ),
                session_id=session_id,
            )
        recorded = await _record(user_id, name, use_case, session_id)
        return IntegrationRequestResponse(
            service=name,
            recorded=recorded,
            message=_message(name, recorded),
            session_id=session_id,
        )


async def _record(
    user_id: str, service: str, use_case: str, session_id: str | None
) -> bool:
    """Add this ask to the service's feature request, filing it if new."""
    title = f"{TITLE_PREFIX}{service}"
    description = (use_case or "").strip()[:MAX_USE_CASE_CHARS]
    filed = await file_feature_request(
        user_id,
        title,
        description or f"A user wants to connect {service}.",
        await find_feature_request(title),
        session_id,
    )
    if filed.type == ResponseType.FEATURE_REQUEST_CREATED:
        return True
    logger.warning(f"Integration request for {service} was not recorded")
    return False


def _message(service: str, recorded: bool) -> str:
    noted = " and that their request is recorded" if recorded else ""
    return (
        f"{service} is not supported yet. The card already tells the user "
        f"so{noted}; do not repeat it. Go straight to what you can do with "
        "the services they can connect. Offer a hand-built HTTP request only "
        f"if they ask for a workaround, and never as a way to sign in to "
        f"{service}."
    )
