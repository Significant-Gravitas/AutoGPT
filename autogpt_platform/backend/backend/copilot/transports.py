"""Chat transport discovery, and the connection a user made their default.

Transport enumeration used to live in ``api.features.chat.routes``, which
worked while the only caller was an HTTP request. It no longer is: bot links,
schedules, briefings and dream passes all open chats with no request to read a
route from, and none of them can import the API layer. So the enumeration lives
here and the routes module presents it.

The default is stored as a provider *and* a credential id. Provider alone stops
being an answer the moment a user can link more than one ChatGPT account, and
this contract is the one the UI will keep.
"""

import logging
from typing import Optional, get_args

from pydantic import BaseModel

from backend.copilot.config import ChatConfig, CopilotLlmAuthProvider
from backend.data.model import Credentials
from backend.data.user import get_user_default_chat_route, set_user_default_chat_route
from backend.integrations.codex.access import has_codex_access_for_discovery
from backend.integrations.codex.auth_bundle import CodexAuthBundleError
from backend.integrations.codex.credential_codec import bundle_from_credentials
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.settings import BehaveAs, Settings

logger = logging.getLogger(__name__)

settings = Settings()
config = ChatConfig()
credentials_manager = IntegrationCredentialsManager()

KNOWN_AUTH_PROVIDERS: frozenset[str] = frozenset(get_args(CopilotLlmAuthProvider))


class ChatTransportResponse(BaseModel):
    auth_provider: CopilotLlmAuthProvider
    credential_id: str | None
    label: str
    available: bool
    default: bool


class ChatTransportsResponse(BaseModel):
    transports: list[ChatTransportResponse]


class DefaultChatRoute(BaseModel):
    """A saved default. ``auth_provider=None`` means automatic."""

    auth_provider: CopilotLlmAuthProvider | None = None
    credential_id: str | None = None


class InvalidDefaultChatRoute(ValueError):
    """A default that can't be saved.

    ``detail`` reuses the codes ``POST /sessions`` already returns for the same
    mistakes, so a client only has to understand one vocabulary.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def is_deployment_chat_available() -> bool:
    if settings.config.behave_as == BehaveAs.CLOUD:
        return True
    api_key, _ = config.main_client_credentials
    return bool(config.test_mode or config.use_claude_code_subscription or api_key)


async def get_chat_transports(user_id: str) -> list[ChatTransportResponse]:
    """Every transport this user can chat over, one of them marked default."""
    transports = [
        ChatTransportResponse(
            auth_provider="platform",
            credential_id=None,
            label=(
                "AutoGPT Platform"
                if settings.config.behave_as == BehaveAs.CLOUD
                else "Self-hosted chat"
            ),
            available=is_deployment_chat_available(),
            default=False,
        )
    ]
    transports.extend(
        ChatTransportResponse(
            auth_provider="codex",
            credential_id=credentials.id,
            label="ChatGPT",
            available=True,
            default=False,
        )
        for credentials in await _valid_codex_credentials(user_id)
    )

    saved_provider, saved_credential_id = await get_user_default_chat_route(user_id)
    _mark_default(transports, saved_provider, saved_credential_id)
    return transports


async def resolve_default_chat_route(
    user_id: str,
) -> tuple[CopilotLlmAuthProvider, str | None]:
    """The route for a chat nobody routed — bot links, schedules, briefings.

    Never raises and never asks. Where the HTTP path can answer "pick one"
    with a 409, these callers have no user in front of them, so an
    unresolvable default falls back to ``platform`` — which is exactly what
    every one of them passed before this setting existed.
    """
    try:
        transports = await get_chat_transports(user_id)
    except Exception:
        logger.warning(
            "Could not resolve the default chat route for user ...%s; using platform",
            user_id[-8:],
            exc_info=True,
        )
        return "platform", None

    default = next((transport for transport in transports if transport.default), None)
    if default is None:
        return "platform", None
    return default.auth_provider, default.credential_id


async def save_default_chat_route(
    user_id: str, route: DefaultChatRoute
) -> list[ChatTransportResponse]:
    """Validate and persist a default, returning the refreshed transport list."""
    if route.auth_provider is None:
        if route.credential_id is not None:
            raise InvalidDefaultChatRoute("codex_credential_not_allowed")
        await set_user_default_chat_route(user_id, None, None)
        return await get_chat_transports(user_id)

    if route.auth_provider == "platform" and route.credential_id is not None:
        raise InvalidDefaultChatRoute("codex_credential_not_allowed")
    if route.auth_provider == "codex" and route.credential_id is None:
        raise InvalidDefaultChatRoute("codex_credential_required")

    transports = await get_chat_transports(user_id)
    if _find_transport(transports, route.auth_provider, route.credential_id) is None:
        # A codex credential the user doesn't own, or can't use on their plan,
        # is indistinguishable from one that doesn't exist — deliberately, so
        # this never confirms another user's credential id.
        raise InvalidDefaultChatRoute(
            "codex_credential_not_found"
            if route.auth_provider == "codex"
            else "chat_transport_not_configured"
        )

    await set_user_default_chat_route(user_id, route.auth_provider, route.credential_id)
    _mark_default(transports, route.auth_provider, route.credential_id)
    return transports


def _mark_default(
    transports: list[ChatTransportResponse],
    saved_provider: Optional[str],
    saved_credential_id: Optional[str],
) -> None:
    chosen = _saved_default(
        transports, saved_provider, saved_credential_id
    ) or _automatic_default(transports)
    for transport in transports:
        transport.default = transport is chosen


def _saved_default(
    transports: list[ChatTransportResponse],
    saved_provider: Optional[str],
    saved_credential_id: Optional[str],
) -> ChatTransportResponse | None:
    """The saved choice, or None when it can no longer be honoured.

    Falling through to the automatic default is the heal: a disconnected
    account or a lapsed plan quietly stops being the default rather than
    failing a Discord message with a reason nobody can see. The row is left
    in place, so reconnecting restores the choice.
    """
    if saved_provider is None:
        return None
    if saved_provider not in KNOWN_AUTH_PROVIDERS:
        # A value written by a newer server. Treat it as automatic rather than
        # letting it break an older one mid-rollout.
        logger.warning("Ignoring unknown saved chat transport %r", saved_provider)
        return None
    return _find_transport(transports, saved_provider, saved_credential_id)


def _automatic_default(
    transports: list[ChatTransportResponse],
) -> ChatTransportResponse | None:
    """What the server picked before anyone could save a preference."""
    available = [transport for transport in transports if transport.available]
    platform = next(
        (transport for transport in available if transport.auth_provider == "platform"),
        None,
    )
    if platform is not None:
        return platform
    codex = [transport for transport in available if transport.auth_provider == "codex"]
    return codex[0] if len(codex) == 1 else None


def _find_transport(
    transports: list[ChatTransportResponse],
    auth_provider: str,
    credential_id: str | None,
) -> ChatTransportResponse | None:
    return next(
        (
            transport
            for transport in transports
            if transport.available
            and transport.auth_provider == auth_provider
            and transport.credential_id == credential_id
        ),
        None,
    )


async def _valid_codex_credentials(user_id: str) -> list[Credentials]:
    if not await has_codex_access_for_discovery(user_id):
        return []
    credentials = await credentials_manager.store.get_creds_by_provider(
        user_id, "codex"
    )
    return [
        credential
        for credential in credentials
        if _is_valid_codex_credentials(credential)
    ]


def _is_valid_codex_credentials(credentials: Credentials | None) -> bool:
    if credentials is None or credentials.type != "oauth2":
        return False
    try:
        bundle_from_credentials(credentials)
    except CodexAuthBundleError:
        return False
    return True
