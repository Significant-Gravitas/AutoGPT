import json
import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from backend.copilot.config import CopilotLLMModel, CopilotMode
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    maybe_append_user_message,
    upsert_chat_session,
)
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamFinishStep,
    StreamStart,
    StreamStartStep,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
)
from backend.copilot.service import strip_user_context_tags
from backend.copilot.token_tracking import persist_and_record_usage
from backend.data.model import OAuth2Credentials
from backend.data.user import get_user_by_id
from backend.integrations.credential_lease import CredentialLease
from backend.integrations.oauth.microsoft_365_copilot import (
    Microsoft365CopilotDeviceAuthHandler,
)
from backend.integrations.providers import ProviderName
from backend.util.exceptions import NotFoundError
from backend.util.timezone_utils import get_user_timezone_or_utc

from .client import Microsoft365CopilotClient, Microsoft365CopilotError

logger = logging.getLogger(__name__)

_MODEL_NAME = "microsoft-365-copilot"
_MAX_ADDITIONAL_CONTEXT_CHARS = 20_000


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _build_additional_context(context: dict[str, str] | None) -> list[str] | None:
    if not context:
        return None
    provider_context = {
        key: value
        for key, value in context.items()
        if key not in {"microsoft_365_web_enabled", "microsoft_365_file_uris"}
    }
    if not provider_context:
        return None
    serialized = json.dumps(provider_context, ensure_ascii=False, default=str)
    return [serialized[:_MAX_ADDITIONAL_CONTEXT_CHARS]]


def _web_enabled(context: dict[str, str] | None) -> bool | None:
    raw_value = context.get("microsoft_365_web_enabled") if context else None
    if raw_value is None:
        return None
    return str(raw_value).lower() not in {"0", "false", "no", "off"}


def _file_uris(context: dict[str, str] | None) -> list[str] | None:
    raw_value = context.get("microsoft_365_file_uris") if context else None
    if not raw_value:
        return None
    try:
        parsed = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, list):
        return None
    uris = [value for value in parsed if isinstance(value, str) and value]
    return uris or None


async def _get_timezone(user_id: str | None) -> str:
    if not user_id:
        return "UTC"
    try:
        user = await get_user_by_id(user_id)
    except Exception:
        logger.warning("Could not load user timezone for Microsoft 365 Copilot")
        return "UTC"
    return get_user_timezone_or_utc(user.timezone)


def _validate_credential_lease(credential_lease: CredentialLease | None) -> str:
    if credential_lease is None:
        raise RuntimeError("microsoft_365_copilot_credential_required")
    credentials = credential_lease.credentials
    required_scopes = set(Microsoft365CopilotDeviceAuthHandler.CHAT_SCOPES)
    if (
        not isinstance(credentials, OAuth2Credentials)
        or credentials.provider != ProviderName.MICROSOFT_365_COPILOT
        or not required_scopes.issubset(credentials.scopes)
    ):
        raise RuntimeError("microsoft_365_copilot_credential_not_found")
    return credentials.access_token.get_secret_value()


def _conversation_key(session: ChatSession) -> str:
    credential_id = session.metadata.llm_credential_id or "unknown"
    return f"microsoft_365_copilot:{credential_id}"


async def stream_chat_completion_microsoft_365(
    session_id: str,
    message: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    session: ChatSession | None = None,
    file_ids: list[str] | None = None,
    context: dict[str, str] | None = None,
    mode: CopilotMode | None = None,
    model: CopilotLLMModel | None = None,
    credential_lease: CredentialLease | None = None,
    **_kwargs: Any,
) -> AsyncGenerator[StreamBaseResponse, None]:
    del mode, model
    if session is None:
        session = await get_chat_session(session_id, user_id)
    if not session:
        raise NotFoundError(
            f"Session {session_id} not found. Please create a new session first."
        )
    if session.metadata.llm_auth_provider != "microsoft_365_copilot":
        raise RuntimeError("microsoft_365_copilot_session_route_mismatch")
    if file_ids:
        yield StreamError(
            errorText=(
                "Microsoft 365 Copilot can only attach OneDrive or SharePoint "
                "links, not local workspace files."
            ),
            code="microsoft_365_copilot_local_files_unsupported",
        )
        return

    access_token = _validate_credential_lease(credential_lease)
    sanitized_message = strip_user_context_tags(message or "").strip()
    if not sanitized_message:
        yield StreamError(
            errorText="Microsoft 365 Copilot requires a text message.",
            code="microsoft_365_copilot_message_required",
        )
        return
    maybe_append_user_message(session, sanitized_message, is_user_message)

    message_id = str(uuid4())
    text_id = f"{message_id}-text"
    response_text = ""
    timezone = await _get_timezone(user_id)

    try:
        async with Microsoft365CopilotClient(access_token) as client:
            conversation_key = _conversation_key(session)
            conversation_id = session.metadata.llm_provider_session_ids.get(
                conversation_key
            )
            if not conversation_id:
                conversation_id = await client.create_conversation()
                session.metadata.llm_provider_session_ids[conversation_key] = (
                    conversation_id
                )
                session = await upsert_chat_session(session)

            yield StreamStart(messageId=message_id, sessionId=session_id)
            yield StreamStartStep()
            yield StreamTextStart(id=text_id)
            async for delta in client.stream_chat(
                conversation_id,
                sanitized_message,
                timezone=timezone,
                additional_context=_build_additional_context(context),
                web_enabled=_web_enabled(context),
                file_uris=_file_uris(context),
            ):
                response_text += delta
                yield StreamTextDelta(id=text_id, delta=delta)
    except Microsoft365CopilotError as error:
        yield StreamError(
            errorText=str(error),
            code="microsoft_365_copilot_request_failed",
        )
        return

    session.messages.append(
        ChatMessage(role="assistant", content=response_text, model=_MODEL_NAME)
    )
    await persist_and_record_usage(
        session=session,
        user_id=user_id,
        prompt_tokens=_estimate_tokens(sanitized_message),
        completion_tokens=_estimate_tokens(response_text),
        log_prefix="[Microsoft365Copilot]",
        cost_usd=None,
        model=_MODEL_NAME,
        provider=ProviderName.MICROSOFT_365_COPILOT.value,
        credential_id_override=session.metadata.llm_credential_id,
        extra_metadata={"billing_mode": "user_subscription"},
    )
    await upsert_chat_session(session)

    yield StreamTextEnd(id=text_id)
    yield StreamFinishStep()
    yield StreamFinish()
