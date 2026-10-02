"""Choose, while a schedule is being made, the accounts its turns will use.

A scheduled turn runs with nobody watching, so it cannot ask which of two
accounts to use, and taking the first saved one silently runs a user's work on
whichever key they happened to add first (SECRT-2804). The chat that makes the
schedule is the one moment someone can answer. So the schedule tools ask there,
with the same card a block run uses, and store each choice on the schedule as a
pin that every one of its turns runs on.

What a schedule will need cannot be read off its prompt, which is free text.
Two things are known: the integrations the model says the work uses, and the
accounts the user already picked in this chat. Both are pinned; a provider
with a single account of the user's own needs no pin, since there is nothing
to choose between.
"""

import logging
from typing import Any

from backend.copilot.context import is_unattended_turn
from backend.copilot.credential_selection import (
    CredentialPin,
    CredentialPins,
    selected_credentials,
)
from backend.copilot.model import ChatSession
from backend.data.model import Credentials, CredentialsFieldInfo, CredentialsType
from backend.integrations.credentials_store import is_system_credential
from backend.integrations.providers import ProviderName

from .expert_scope import provider_slug
from .models import SetupInfo, SetupRequirementsResponse, UserReadiness
from .utils import build_missing_credentials_from_field_info, get_user_credentials

logger = logging.getLogger(__name__)

INTEGRATIONS_PARAM: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string"},
    "description": (
        "Providers the work uses, e.g. ['exa']. If the user has several "
        "accounts for one, a card asks which; call again once they pick."
    ),
}


async def has_account_choices(
    session: ChatSession, integrations: list[str] | None
) -> bool:
    """Whether this call names an integration or this chat picked an account,
    i.e. whether there is anything for ``pin_schedule_credentials`` to do."""
    return bool(_providers(integrations)) or bool(
        await selected_credentials(session.session_id)
    )


async def pin_schedule_credentials(
    user_id: str,
    session: ChatSession,
    expert_id: str | None,
    integrations: list[str] | None,
    existing: CredentialPins | None = None,
) -> CredentialPins | SetupRequirementsResponse:
    """The pins to store on a schedule, or the card asking for a choice.

    *expert_id* is the scope the schedule runs in, so only accounts its turns
    could use are offered. *existing* is what a routine being changed already
    holds; a pick made in this chat replaces it, which is how the user moves a
    routine to another account.
    """
    pins: CredentialPins = dict(existing or {})
    picked = await selected_credentials(session.session_id)
    providers = _providers(integrations)
    if not picked and not providers:
        return pins
    own = [
        c
        for c in await get_user_credentials(user_id, expert_id)
        if not is_system_credential(c.id)
    ]
    by_id = {c.id: c for c in own}

    for provider, credential_id in picked.items():
        cred = by_id.get(credential_id)
        if cred is not None and provider_slug(cred.provider) == provider:
            pins[provider] = CredentialPin(id=cred.id, title=cred.title or "")

    choices: dict[str, list[Credentials]] = {}
    for provider in providers:
        pin = pins.get(provider)
        if pin is not None and pin.id in by_id:
            continue
        fits = [c for c in own if provider_slug(c.provider) == provider]
        if len(fits) > 1:
            choices[provider] = fits
        else:
            # A pin whose account is gone, now with one account or none left:
            # nothing to choose, so the turn takes what there is.
            pins.pop(provider, None)

    if not choices:
        return pins
    if is_unattended_turn():
        # A scheduled turn scheduling more work has nobody to ask either.
        logger.info(
            "Unattended turn in session %s scheduled work using %s without "
            "pinning an account; its turns will use the first saved one",
            session.session_id,
            ", ".join(sorted(choices)),
        )
        return pins
    return _choice_card(session.session_id, choices)


def _providers(integrations: list[str] | None) -> list[str]:
    seen: list[str] = []
    for raw in integrations or []:
        provider = str(raw).strip().lower()
        if provider and provider not in seen:
            seen.append(provider)
    return seen


def _choice_card(
    session_id: str, choices: dict[str, list[Credentials]]
) -> SetupRequirementsResponse:
    fields = {
        f"{provider}_credentials": CredentialsFieldInfo[ProviderName, CredentialsType](
            credentials_provider=frozenset([ProviderName(provider)]),
            credentials_types=frozenset(c.type for c in fits),
            credentials_scopes=None,
        )
        for provider, fits in choices.items()
    }
    missing = build_missing_credentials_from_field_info(fields, set())
    accounts = "; ".join(
        f"{provider}: " + ", ".join(repr(c.title or c.id) for c in fits)
        for provider, fits in choices.items()
    )
    return SetupRequirementsResponse(
        message=(
            "Nothing is scheduled yet. The user has more than one account for "
            f"{', '.join(sorted(choices))} ({accounts}), and nobody will be "
            "there to ask when it runs. Ask them which account this schedule "
            "should use with the card below, then call this tool again with "
            "the same arguments; every run will use the account they pick."
        ),
        session_id=session_id,
        setup_info=SetupInfo(
            agent_id="schedule_accounts",
            agent_name="Scheduled run",
            user_readiness=UserReadiness(
                has_all_credentials=False,
                missing_credentials=missing,
                ready_to_run=False,
            ),
            requirements={
                "credentials": list(missing.values()),
                "inputs": [],
                "execution_modes": [],
            },
        ),
    )
