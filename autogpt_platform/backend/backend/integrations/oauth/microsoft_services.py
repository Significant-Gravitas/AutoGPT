"""What a Microsoft connection can be used for, and the permissions behind it."""

from typing import Literal

from backend.data.model import Credentials, OAuth2Credentials
from backend.integrations.oauth.microsoft_365_copilot import (
    Microsoft365CopilotDeviceAuthHandler,
)
from backend.integrations.oauth.microsoft_identity import microsoft_identity
from backend.integrations.providers import ProviderName

GRAPH_PREFIX = "https://graph.microsoft.com/"
WORK_IQ_SCOPE = "api://workiq.svc.cloud.microsoft/WorkIQAgent.Ask"
COPILOT_SCOPES = list(Microsoft365CopilotDeviceAuthHandler.CHAT_SCOPES)
SIGN_IN_SCOPES = ["openid", "profile", "email", "offline_access"]

MicrosoftService = Literal["mail", "calendar", "files", "teams", "copilot", "work_iq"]
MICROSOFT_SERVICE_SCOPES: dict[MicrosoftService, list[str]] = {
    "mail": ["Mail.ReadWrite", "Mail.Send"],
    "calendar": ["Calendars.ReadWrite"],
    "files": ["Files.ReadWrite"],
    "teams": ["Chat.ReadWrite"],
    "copilot": COPILOT_SCOPES,
    "work_iq": [WORK_IQ_SCOPE],
}
# Microsoft offers these only to work and school accounts; asking a personal
# account for them fails the whole sign-in.
WORK_ONLY_SERVICES: frozenset[MicrosoftService] = frozenset(
    {"teams", "copilot", "work_iq"}
)
DEFAULT_MICROSOFT_SCOPES = [
    "User.Read",
    *MICROSOFT_SERVICE_SCOPES["mail"],
    *MICROSOFT_SERVICE_SCOPES["calendar"],
    *MICROSOFT_SERVICE_SCOPES["files"],
]


def consent_scopes(scopes: list[str]) -> list[str]:
    """Everything a sign-in asks consent for. May span Graph and Work IQ."""
    requested = scopes or DEFAULT_MICROSOFT_SCOPES
    return list(dict.fromkeys([*SIGN_IN_SCOPES, "User.Read", *requested]))


def token_scopes(scopes: list[str]) -> list[str]:
    """A token request names one resource. Ours is always Graph."""
    graph = [
        scope.removeprefix(GRAPH_PREFIX)
        for scope in scopes
        if not scope.startswith("api://")
    ]
    return list(dict.fromkeys([*SIGN_IN_SCOPES, *graph]))


def is_microsoft_365_copilot_credential(credentials: Credentials | None) -> bool:
    """A connection the Microsoft 365 Copilot chat route can run on.

    Either a Microsoft work account that has granted Copilot's permissions, or
    a Copilot connection made before Microsoft connections were combined.
    """
    if not isinstance(credentials, OAuth2Credentials) or not set(
        COPILOT_SCOPES
    ).issubset(credentials.scopes):
        return False
    if credentials.provider == ProviderName.MICROSOFT_365_COPILOT:
        return True
    identity = microsoft_identity(credentials)
    return (
        credentials.provider == ProviderName.MICROSOFT
        and identity is not None
        and identity.account_type == "work"
    )
