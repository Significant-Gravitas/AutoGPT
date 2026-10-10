"""AutoPilot's side of the managed Ayrshare profile (SECRT-2878).

Ayrshare blocks post with a per-user profile key that AutoGPT provisions
(:class:`~backend.integrations.managed_providers.ayrshare.AyrshareManagedProvider`).
It opts out of the credentials sweep, so a user who never opened the
builder's "Connect Social Media Accounts" button has no credential, and the
copilot used to answer every ``post_to_*`` call with a card asking for an
API key the user cannot have.  Here the copilot provisions the profile on
demand through the same path as ``/ayrshare/sso_url`` and hands back the SSO
link where the user links their networks.
"""

import logging
import re
from typing import Any

from backend.blocks._base import AnyBlockSchema
from backend.copilot.context import is_unattended_turn
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.managed_providers.ayrshare import (
    AyrshareNotConfiguredError,
    AyrshareProvisioningError,
    AyrshareSSOLink,
    get_sso_link,
)

from .expert_scope import provider_slug
from .models import (
    ErrorResponse,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)
from .utils import build_missing_credentials_from_field_info, sanitize_provider_message

logger = logging.getLogger(__name__)

AYRSHARE = "ayrshare"

NO_API_KEY = (
    "AutoGPT holds the Ayrshare account, so never ask the user for an Ayrshare "
    "API key or subscription."
)

NOT_AVAILABLE_MESSAGE = (
    "Posting to social networks through Ayrshare is not available on this "
    "AutoGPT deployment (it has not been set up here). Tell the user plainly "
    "that social posting is not available yet. " + NO_API_KEY
)

PROVISIONING_FAILED_MESSAGE = (
    "Could not set up the user's social posting profile just now. Try again "
    "in a moment. " + NO_API_KEY
)

# Ayrshare answers a post to a network the profile has not linked with a
# per-platform error such as "Twitter is not linked" or "... Please confirm
# the linkage on the Social Accounts page".
_NOT_LINKED_RE = re.compile(
    r"not\s+linked|linkage|no\s+(linked\s+)?social\s+accounts?", re.IGNORECASE
)


def is_ayrshare(provider: object) -> bool:
    return provider_slug(provider).strip().lower() == AYRSHARE


def is_not_linked_error(exc: BaseException) -> bool:
    """True when Ayrshare refused a post because the network is not linked."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _NOT_LINKED_RE.search(str(current)):
            return True
        current = current.__cause__ or current.__context__
    return False


async def sso_link(user_id: str) -> AyrshareSSOLink | ErrorResponse:
    """The user's SSO link, provisioning their profile first if needed, or
    the error the copilot should relay instead."""
    try:
        return await get_sso_link(user_id, IntegrationCredentialsManager().store)
    except AyrshareNotConfiguredError:
        return ErrorResponse(
            message=NOT_AVAILABLE_MESSAGE, error="ayrshare_not_configured"
        )
    except AyrshareProvisioningError as exc:
        logger.warning("Ayrshare SSO link failed for user %s: %s", user_id, exc)
        return ErrorResponse(
            message=PROVISIONING_FAILED_MESSAGE, error="ayrshare_provisioning_failed"
        )


def link_message(link: AyrshareSSOLink, lead: str) -> str:
    return (
        f"{lead} Give the user this link to link their social accounts on "
        f"Ayrshare's page (valid until {link.expires_at:%Y-%m-%d %H:%M} UTC): "
        f"{link.url} — or the Connect Social Media Accounts button in the card. "
        "Once they say they have linked them, run the step again. "
        "Each post costs 2 credits (5 for video). " + NO_API_KEY
    )


def link_accounts_card(
    *,
    agent_id: str,
    agent_name: str,
    missing_credentials: dict[str, dict[str, Any]],
    link: AyrshareSSOLink,
    message: str,
    session_id: str | None,
    execution_modes: tuple[str, ...] = ("immediate",),
) -> SetupRequirementsResponse:
    """The setup card for an Ayrshare field.

    Each entry keeps the usual shape (provider ``ayrshare``), which the chat
    card renders as the Ayrshare connect button rather than a key field, and
    carries the SSO link as ``connect_url`` for a card that opens it directly.
    """
    entries = {
        key: {**entry, "connect_url": link.url} for key, entry in missing_credentials.items()
    }
    return SetupRequirementsResponse(
        message=message,
        session_id=session_id,
        setup_info=SetupInfo(
            agent_id=agent_id,
            agent_name=agent_name,
            user_readiness=UserReadiness(
                has_all_credentials=False,
                missing_credentials=entries,
                ready_to_run=False,
            ),
            requirements={
                "credentials": list(entries.values()),
                "inputs": [],
                "execution_modes": list(execution_modes),
            },
        ),
    )


async def provision_for_block(
    user_id: str, block: AnyBlockSchema, block_id: str, session_id: str | None
) -> ToolResponseBase | None:
    """Answer for an Ayrshare block that found no credential to run with.

    Provisions the user's managed profile and returns the link card, since a
    profile that was just created has no linked networks yet.  ``None`` when
    the user already has the managed credential and something else kept it
    out of the match (an expert grant, a pinned account): the usual card
    handles those.
    """
    store = IntegrationCredentialsManager().store
    if await store.has_managed_credential(user_id, AYRSHARE):
        return None
    reason = "no social account is linked to the user's AutoGPT posting profile yet"
    return await _block_link_answer(
        user_id,
        block,
        block_id,
        session_id,
        reason=reason,
        lead=(
            f"Block '{block.name}' posts through AutoGPT's managed Ayrshare "
            f"account, and {reason}."
        ),
    )


async def connect_for_block(
    user_id: str, block: AnyBlockSchema, block_id: str, session_id: str | None
) -> ToolResponseBase:
    """Answer for an explicit connect on an Ayrshare block.

    Having the managed credential says nothing about which networks are
    linked, so connecting always means handing over the link.
    """
    return await _block_link_answer(
        user_id,
        block,
        block_id,
        session_id,
        reason="the user asked to connect their social accounts",
        lead=f"Block '{block.name}' posts through AutoGPT's managed Ayrshare account.",
    )


async def not_linked_response(
    user_id: str,
    block: AnyBlockSchema,
    block_id: str,
    exc: BaseException,
    session_id: str | None,
) -> ToolResponseBase:
    """Answer for a post Ayrshare refused because the network is not linked."""
    detail = sanitize_provider_message(str(exc))
    return await _block_link_answer(
        user_id,
        block,
        block_id,
        session_id,
        reason=f"the network is not linked to the user's posting profile ({detail})",
        lead=f"Block '{block.name}' did not post: Ayrshare said \"{detail}\".",
        outside=(detail,),
    )


async def _block_link_answer(
    user_id: str,
    block: AnyBlockSchema,
    block_id: str,
    session_id: str | None,
    *,
    reason: str,
    lead: str,
    outside: tuple[str, ...] = (),
) -> ToolResponseBase:
    link = await sso_link(user_id)
    if isinstance(link, ErrorResponse):
        link.session_id = session_id
        return link.from_outside()
    if is_unattended_turn():
        return _unattended_link_error(
            f"Block '{block.name}'", reason, link, session_id
        ).from_outside(*outside)
    return link_accounts_card(
        agent_id=block_id,
        agent_name=block.name,
        missing_credentials=_block_entries(block),
        link=link,
        message=link_message(link, lead),
        session_id=session_id,
    ).from_outside(*outside)


def _unattended_link_error(
    subject: str, reason: str, link: AyrshareSSOLink, session_id: str | None
) -> ErrorResponse:
    logger.warning(
        "Unattended copilot turn in session %s: %s skipped, %s",
        session_id,
        subject,
        reason,
    )
    return ErrorResponse(
        message=(
            f"{subject} did not post: {reason}. Nobody is watching this turn "
            "(it was scheduled), so say plainly in your reply that the post "
            "was skipped and that the user can link their social accounts "
            f"here: {link.url} . " + NO_API_KEY
        ),
        error="ayrshare_accounts_not_linked",
        session_id=session_id,
    )


def _block_entries(block: AnyBlockSchema) -> dict[str, dict[str, Any]]:
    return build_missing_credentials_from_field_info(
        {
            field: info
            for field, info in block.input_schema.get_credentials_fields_info().items()
            if any(is_ayrshare(p) for p in info.provider)
        },
        matched_keys=set(),
    )
