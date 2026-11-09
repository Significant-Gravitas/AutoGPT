"""Why the user's own credential does not satisfy a requirement.

A setup card used to say "connect your account" whether the user had never
connected it, had connected it without a scope the block needs, or had a
credential whose refresh token the provider refused for good. The model and
the user both read that as "not connected", reconnected with the same narrow
grant, and looped. This module tells the three apart so the card can name the
missing scopes or the reason to reconnect, and so the reconnect it starts asks
for every scope that is needed rather than only the one that was missing.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel

from backend.data.model import Credentials, CredentialsFieldInfo, OAuth2Credentials
from backend.integrations.credentials_store import is_system_credential
from backend.integrations.oauth.refresh_failure import (
    provider_display_name,
    reconnect_required,
)

from .utils import find_matching_credential, get_user_credentials

logger = logging.getLogger(__name__)


class CredentialGap(BaseModel):
    kind: Literal["missing_scopes", "reconnect_required"]
    credential_id: str
    credential_title: str | None = None
    missing_scopes: list[str] = []
    granted_scopes: list[str] = []
    reason: str | None = None


async def credentials_for_gaps(
    user_id: str, expert_id: str | None
) -> list[Credentials]:
    """The user's credentials, or none if they cannot be read right now.

    They only sharpen a card's wording; a failed read must not cost the user
    the card itself.
    """
    try:
        return await get_user_credentials(user_id, expert_id)
    except Exception:
        logger.warning("Could not read credentials to explain a card", exc_info=True)
        return []


def find_credential_gap(
    available: list[Credentials], field_info: CredentialsFieldInfo
) -> CredentialGap | None:
    """The closest of the user's own OAuth credentials and what it lacks.

    ``None`` when a healthy credential already satisfies *field_info*, or when
    the user has no OAuth credential for the provider at all (never connected).
    """
    if "oauth2" not in field_info.supported_types:
        return None
    own = [
        c
        for c in available
        if isinstance(c, OAuth2Credentials)
        and not is_system_credential(c.id)
        and c.provider in field_info.provider
    ]
    if not own:
        return None
    fitting = [c for c in own if find_matching_credential([c], field_info)]
    if any(reconnect_required(c) is None for c in fitting):
        return None
    if fitting:
        return _reconnect_gap(fitting[0])
    return _scope_gap(own, field_info)


def _reconnect_gap(credential: OAuth2Credentials) -> CredentialGap:
    marker = reconnect_required(credential)
    return CredentialGap(
        kind="reconnect_required",
        credential_id=credential.id,
        credential_title=credential.title or credential.username,
        granted_scopes=sorted(credential.scopes),
        reason=(
            marker.reason(provider_display_name(credential.provider))
            if marker
            else None
        ),
    )


def _scope_gap(
    own: list[OAuth2Credentials], field_info: CredentialsFieldInfo
) -> CredentialGap | None:
    required = set(field_info.required_scopes or ())
    if not required:
        return None
    # The credential that is closest to fitting is the one worth upgrading.
    closest = min(own, key=lambda c: len(required - set(c.scopes)))
    missing = required - set(closest.scopes)
    if not missing:
        # It has the scopes but fails on something else (host, server URL);
        # that is not a gap this card can explain.
        return None
    marker = reconnect_required(closest)
    return CredentialGap(
        kind="missing_scopes",
        credential_id=closest.id,
        credential_title=closest.title or closest.username,
        missing_scopes=sorted(missing),
        granted_scopes=sorted(closest.scopes),
        reason=(
            marker.reason(provider_display_name(closest.provider)) if marker else None
        ),
    )


def apply_credential_gap(entry: dict[str, Any], gap: CredentialGap) -> dict[str, Any]:
    """*entry* (a serialized missing credential) annotated with *gap*.

    The entry's ``scopes`` are what the card's reconnect requests, so they
    become the union of what the requirement needs and what the credential
    already has: requesting only the missing scope would come back as a grant
    narrower than the one it replaces.
    """
    scopes = sorted(set(entry.get("scopes") or []) | set(gap.granted_scopes))
    return {**entry, "scopes": scopes, "credential_gap": gap.model_dump()}


def credential_gap_message(provider_name: str, gap: CredentialGap) -> str:
    named = f" '{gap.credential_title}'" if gap.credential_title else ""
    if gap.kind == "missing_scopes":
        plural = "s" if len(gap.missing_scopes) > 1 else ""
        message = (
            f"Your {provider_name} account{named} is connected, but it was not "
            f"granted the {', '.join(gap.missing_scopes)} permission{plural} "
            f"this needs. Reconnect it and approve "
            f"{'those permissions' if plural else 'that permission'}."
        )
        if gap.reason:
            message += f" {gap.reason}."
        return message
    reason = f": {gap.reason}" if gap.reason else ""
    return (
        f"Your saved {provider_name} account{named} has to be reconnected"
        f"{reason}. Reconnect it to continue."
    )


def annotate_credential_gaps(
    available: list[Credentials],
    fields: dict[str, CredentialsFieldInfo],
    missing: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Annotate every entry in *missing* that has a gap, and say what it is.

    Returns the annotated mapping and one user-facing sentence per gap.
    """
    annotated = dict(missing)
    messages: list[str] = []
    for key, entry in missing.items():
        field_info = fields.get(key)
        if field_info is None:
            continue
        gap = find_credential_gap(available, field_info)
        if gap is None:
            continue
        annotated[key] = apply_credential_gap(entry, gap)
        provider_name = str(entry.get("provider_name") or "") or (
            provider_display_name(str(entry.get("provider") or ""))
        )
        messages.append(credential_gap_message(provider_name, gap))
    return annotated, messages
