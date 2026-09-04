"""Codex access over HTTP.

Sign-in, the model catalog, rate limits and inference all run against
``auth.openai.com`` and ``chatgpt.com`` directly. There is no Codex process, so
the machinery that used to exist purely to manage one -- a runtime pool, a
process capacity semaphore, synthetic ``$HOME`` directories and per-credential
exclusivity -- is gone with it.

The most visible consequence is concurrency: a ChatGPT connection used to serve
one chat at a time because a single subprocess owned the credential. Over HTTP
the same credential serves as many concurrent turns as the account allows.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import cast

from backend.data.model import Credentials, OAuth2Credentials
from backend.integrations.codex.credential_codec import (
    bundle_from_credentials,
    checkpoint_credentials_from_bundle,
)
from backend.integrations.codex.device_login import (
    CodexHttpDeviceLogin,
    start_http_device_login,
)
from backend.integrations.codex.http_client import account_snapshot, fetch_models
from backend.integrations.codex.http_session import (
    CodexHttpSession,
    CodexInvocationTimeoutError,
)
from backend.integrations.codex.models import (
    CodexAccountSnapshot,
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
    CodexInvocationResult,
    CodexModelInfo,
    CodexRateLimitsSnapshot,
    CodexStreamEvent,
)
from backend.integrations.credential_lease import CredentialLease
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

EventHandler = Callable[[CodexStreamEvent], Awaitable[None]] | None
ToolHandler = Callable[[CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]]


class CodexTransportError(RuntimeError):
    pass


class CodexTransportOverloadedError(CodexTransportError):
    pass


class CodexCredentialBusyError(CodexTransportError):
    """Kept for callers that still distinguish it.

    No longer raised by acquisition itself: exclusivity was a property of the
    subprocess, not of the account.
    """


class CodexCredentialIntegrityError(CodexTransportError):
    pass


__all__ = [
    "CodexAgentSession",
    "CodexCredentialBusyError",
    "CodexCredentialIntegrityError",
    "CodexHttpDeviceLogin",
    "CodexInvocationTimeoutError",
    "CodexTransport",
    "CodexTransportError",
    "CodexTransportOverloadedError",
    "get_codex_transport",
]


def _validated_codex_credentials(credentials: Credentials) -> OAuth2Credentials:
    if credentials.type != "oauth2":
        raise CodexTransportError("Codex transport requires OAuth credentials")
    oauth_credentials = cast(OAuth2Credentials, credentials)
    # Rejects a credential whose provider state cannot be read, before it is
    # used for anything that costs the user money.
    bundle_from_credentials(oauth_credentials)
    return oauth_credentials


def codex_credentials(
    lease: CredentialLease | CodexCredentialLease,
) -> OAuth2Credentials:
    return _validated_codex_credentials(lease.credentials)


def resolve_invocation_model(
    requested_model: str | None, models: list[CodexModelInfo]
) -> str:
    available = {model.model: model for model in models}
    if requested_model is not None:
        if requested_model not in available:
            raise CodexTransportError(
                f"Codex model is not available for this account: {requested_model}"
            )
        return requested_model
    default = next((model for model in models if model.is_default), None)
    visible = next((model for model in models if not model.hidden), None)
    selected = default or visible or (models[0] if models else None)
    if selected is None:
        raise CodexTransportError("Codex account advertised no available models")
    return selected.model


class CodexAgentSession:
    """An open Codex turn for one credential."""

    def __init__(self, session: CodexHttpSession, credentials: OAuth2Credentials):
        self._session = session
        self._credentials = credentials

    @property
    def credential_id(self) -> str:
        return self._credentials.id

    async def invoke(
        self,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: ToolHandler,
        event_handler: EventHandler = None,
    ) -> CodexInvocationResult:
        result = await self._session.invoke(
            request, dynamic_tools, tool_handler, event_handler
        )
        return result.model_copy(
            update={"resolved_model": result.resolved_model or request.model}
        )

    async def models(self) -> list[CodexModelInfo]:
        return await fetch_models(self._credentials)

    @property
    def rate_limits(self) -> CodexRateLimitsSnapshot | None:
        return self._session.rate_limits

    @property
    def closed(self) -> bool:
        # Nothing is held open between turns over HTTP.
        return False

    @property
    def failure(self) -> BaseException | None:
        return None


class CodexCredentialLease:
    """An immutable Codex credential snapshot plus the session that spends it.

    Mirrors the surface the copilot executor and gateway already use, so
    swapping the process out did not change their call sites. HTTP turns do
    not hold the credentials-manager lock: refresh is serialized before this
    snapshot is created, then concurrent turns can safely share the resulting
    access token.
    """

    def __init__(
        self, credentials: OAuth2Credentials, session: CodexAgentSession
    ) -> None:
        self._credentials = credentials
        self._session = session

    @property
    def credentials(self) -> OAuth2Credentials:
        return self._credentials

    async def models(self) -> list[CodexModelInfo]:
        return await self._session.models()

    async def invoke(
        self,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: ToolHandler,
        event_handler: EventHandler = None,
    ) -> CodexInvocationResult:
        return await self._session.invoke(
            request, dynamic_tools, tool_handler, event_handler
        )

    async def release(self) -> None:
        # Kept for the executor's common lease lifecycle. No lock survives
        # acquire_runtime_lease(), so concurrent HTTP turns never serialize.
        return None


class CodexTransport:
    def __init__(
        self,
        *,
        invocation_timeout_seconds: float = 180,
        copilot_turn_timeout_seconds: float = 21600,
        copilot_tool_timeout_seconds: float = 900,
        login_timeout_seconds: float = 900,
    ) -> None:
        self._invocation_timeout_seconds = invocation_timeout_seconds
        self._copilot_turn_timeout_seconds = copilot_turn_timeout_seconds
        self._copilot_tool_timeout_seconds = copilot_tool_timeout_seconds
        self._login_timeout_seconds = login_timeout_seconds
        # Last quota reading per credential. Process-local on purpose: it is a
        # display value, and paying for an inference call to refresh it would
        # cost the user real quota.
        self._last_rate_limits: dict[str, CodexRateLimitsSnapshot] = {}

    async def start_device_login(self) -> CodexHttpDeviceLogin:
        return await start_http_device_login(self._login_timeout_seconds)

    async def acquire_runtime_lease(
        self, user_id: str, credential_id: str, *, lock_timeout_seconds: float
    ) -> CodexCredentialLease:
        manager = IntegrationCredentialsManager()

        async def load() -> Credentials | None:
            try:
                return await asyncio.wait_for(
                    manager.get(user_id, credential_id),
                    timeout=lock_timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise CodexCredentialBusyError("codex_credential_busy") from None

        credentials = await load()
        if credentials is None:
            raise CodexTransportError("Codex credential not found")
        oauth_credentials = _validated_codex_credentials(credentials)

        if oauth_credentials.refresh_strategy == "provider_runtime":
            # One-time migration from the removed CLI runtime. Hold the legacy
            # lease only long enough to rewrite refresh ownership; never for
            # the inference turn itself.
            try:
                legacy_lease = await asyncio.wait_for(
                    manager.acquire_lease(user_id, credential_id),
                    timeout=lock_timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise CodexCredentialBusyError("codex_credential_busy") from None
            try:
                current = _validated_codex_credentials(legacy_lease.credentials)
                if current.refresh_strategy == "provider_runtime":
                    migrated = checkpoint_credentials_from_bundle(
                        current, bundle_from_credentials(current)
                    )
                    try:
                        await asyncio.wait_for(
                            legacy_lease.checkpoint(migrated),
                            timeout=lock_timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        raise CodexCredentialBusyError(
                            "codex_credential_busy"
                        ) from None
            finally:
                await legacy_lease.release()

            credentials = await load()
            if credentials is None:
                raise CodexTransportError("Codex credential not found")
            oauth_credentials = _validated_codex_credentials(credentials)

        return CodexCredentialLease(
            oauth_credentials, self._session_for(oauth_credentials)
        )

    async def close_runtime_pool(self) -> None:
        """Nothing is pooled; kept so shutdown call sites stay unchanged."""

    async def account(self, lease: CredentialLease) -> CodexAccountSnapshot:
        return account_snapshot(codex_credentials(lease))

    async def rate_limits(self, lease: CredentialLease) -> CodexRateLimitsSnapshot:
        """Report the quota last seen for this credential.

        ChatGPT only reports quota on the headers of an inference response --
        there is no standalone endpoint, and ``/models`` carries none. So this
        answers with what the most recent turn saw, and an empty snapshot (every
        field ``None``) when this worker has not run one yet. That reads as
        "not known yet", which is true; failing the request instead would make a
        never-used connection look broken.
        """
        credentials = codex_credentials(lease)
        return self._last_rate_limits.get(credentials.id, CodexRateLimitsSnapshot())

    def _record_rate_limits(self, session: "CodexAgentSession") -> None:
        seen = session.rate_limits
        if seen is not None:
            self._last_rate_limits[session.credential_id] = seen

    async def models(self, lease: CredentialLease) -> list[CodexModelInfo]:
        return await fetch_models(codex_credentials(lease))

    async def invoke(
        self, lease: CredentialLease, request: CodexInvocationRequest
    ) -> CodexInvocationResult:
        credentials = codex_credentials(lease)
        models = await fetch_models(credentials)
        resolved_model = resolve_invocation_model(request.model, models)
        session = self._session_for(
            credentials, turn_timeout=self._invocation_timeout_seconds
        )
        try:
            result = await session.invoke(
                request.model_copy(update={"model": resolved_model}), [], _no_tools
            )
            return result.model_copy(update={"resolved_model": resolved_model})
        finally:
            self._record_rate_limits(session)

    async def invoke_agent(
        self,
        lease: CredentialLease,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: ToolHandler,
        event_handler: EventHandler = None,
    ) -> CodexInvocationResult:
        session = self._session_for(codex_credentials(lease))
        try:
            return await session.invoke(
                request, dynamic_tools, tool_handler, event_handler
            )
        finally:
            self._record_rate_limits(session)

    @asynccontextmanager
    async def agent_session(
        self, lease: CredentialLease
    ) -> AsyncIterator[CodexAgentSession]:
        session = self._session_for(codex_credentials(lease))
        try:
            yield session
        finally:
            self._record_rate_limits(session)

    async def logout(self, lease: CredentialLease) -> None:
        """OpenAI publishes no revocation endpoint; the caller drops the row."""
        logger.info("Codex logout requested; dropping the stored credential")

    def _session_for(
        self, credentials: OAuth2Credentials, *, turn_timeout: float | None = None
    ) -> CodexAgentSession:
        return CodexAgentSession(
            CodexHttpSession(
                credentials,
                turn_timeout_seconds=turn_timeout or self._copilot_turn_timeout_seconds,
                tool_timeout_seconds=self._copilot_tool_timeout_seconds,
            ),
            credentials,
        )


async def _no_tools(call: CodexDynamicToolCall) -> CodexDynamicToolResult:
    return CodexDynamicToolResult(
        content=f"Tool {call.tool!r} is not available.", success=False
    )


_transport: CodexTransport | None = None
_transport_lock = threading.Lock()


def get_codex_transport() -> CodexTransport:
    global _transport
    if _transport is None:
        with _transport_lock:
            if _transport is None:
                config = Settings().config
                _transport = CodexTransport(
                    invocation_timeout_seconds=config.codex_invocation_timeout_seconds,
                    copilot_turn_timeout_seconds=config.codex_copilot_turn_timeout_seconds,
                    copilot_tool_timeout_seconds=config.codex_copilot_tool_timeout_seconds,
                    login_timeout_seconds=config.codex_login_timeout_seconds,
                )
    return _transport
