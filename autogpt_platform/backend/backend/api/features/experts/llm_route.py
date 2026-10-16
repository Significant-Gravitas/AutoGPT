"""How an expert's pinned AI connection is read back to its owner.

The Expert row stores the pin as raw strings. What the Team page needs is the
pin's name and whether it can still be honoured, and neither belongs in the
row: the name is the transport's, and availability changes with the owner's
linked accounts and plan. Resolved here against the live transport list, once
per read rather than once per expert.
"""

import logging
from typing import cast

from backend.api.features.experts.models import Expert
from backend.copilot.config import CopilotLlmAuthProvider
from backend.copilot.transports import (
    KNOWN_AUTH_PROVIDERS,
    ChatTransportResponse,
    get_chat_transports,
    resolve_pinned_chat_route,
    transport_label,
)

logger = logging.getLogger(__name__)


def known_auth_provider(value: str | None) -> CopilotLlmAuthProvider | None:
    """A stored provider this server can route, else None (account default).

    A value written by a newer server reads as unpinned rather than breaking
    every expert read mid-rollout — the same rule the account default follows.
    """
    if value is None or value not in KNOWN_AUTH_PROVIDERS:
        return None
    return cast(CopilotLlmAuthProvider, value)


async def annotate_llm_routes(user_id: str, experts: list[Expert]) -> list[Expert]:
    """Fill ``llm_route_label`` and ``llm_route_available`` on pinned experts.

    One transport lookup serves the whole list; nothing is fetched when no
    expert is pinned. A failed lookup leaves the defaults in place — a pin
    reads as available — because a transient error must not raise a
    "connection missing" warning on every expert at once.
    """
    if not any(expert.llm_auth_provider is not None for expert in experts):
        return experts
    try:
        transports = await get_chat_transports(user_id)
    except Exception:
        logger.warning(
            "Could not resolve expert chat routes for user ...%s",
            user_id[-8:],
            exc_info=True,
        )
        return experts
    return [await _with_route_state(user_id, expert, transports) for expert in experts]


async def _with_route_state(
    user_id: str, expert: Expert, transports: list[ChatTransportResponse]
) -> Expert:
    if expert.llm_auth_provider is None:
        return expert
    pinned = await resolve_pinned_chat_route(
        user_id,
        expert.llm_auth_provider,
        expert.llm_credential_id,
        transports=transports,
    )
    return expert.model_copy(
        update={
            "llm_route_label": (
                pinned.label
                if pinned is not None
                else transport_label(expert.llm_auth_provider)
            ),
            "llm_route_available": pinned is not None,
        }
    )
