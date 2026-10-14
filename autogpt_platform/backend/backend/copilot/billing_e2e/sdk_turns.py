"""SDK-engine turns: ``stream_chat_completion_sdk`` driving the bundled CLI.

The CLI really runs and really talks to the fake provider, directly or
through the Codex gateway. Stubbed: persistence, locks, transcript storage,
the copilot's own tools and hooks, and per-user flag lookups. Model
resolution, the CLI environment, retries, resume and every cost decision run
as in production.
"""

import shutil
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from openai import AsyncOpenAI
from pydantic import SecretStr

from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatSession
from backend.copilot.sdk import env as sdk_env
from backend.copilot.sdk import openrouter_cost
from backend.copilot.sdk import service as sdk_service
from backend.copilot.transcript import TranscriptDownload
from backend.copilot.tree import root_envelope
from backend.data.llm_registry.llm_models import LLMModel
from backend.data.model import OAuth2Credentials
from backend.integrations.codex.auth_bundle import CodexAuthBundleV1, CodexAuthTokensV1
from backend.integrations.codex.credential_codec import credentials_from_bundle
from backend.integrations.codex.http_session import CodexHttpSession
from backend.integrations.codex.models import CodexModelInfo
from backend.integrations.codex.transport import CodexAgentSession, CodexCredentialLease

from .provider import FakeProvider

CODEX_MODEL = LLMModel.GPT5_6_TERRA.value
CODEX_CREDENTIAL_ID = "codex-credential"


class SdkTurns:
    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        provider: FakeProvider,
        config: ChatConfig,
        *,
        codex: bool = False,
        cli_max_retries: int | None = None,
    ) -> None:
        self._config_dir = tmp_path / "home" / ".claude"
        self._cwd = tmp_path / "copilot-cwd"
        self._cwd.mkdir()
        self._config = config
        self._transcripts: dict[str, TranscriptDownload] = {}
        self._lease = _codex_lease(provider) if codex else None
        # The turn of each launch, and the CLI session it resumed (None for a
        # fresh one).
        self.resumed: list[tuple[int, str | None]] = []
        self._turn = -1
        # The CLI inherits these, and the platform reads the CLI's session
        # files from the same place.
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(self._config_dir))
        # Where a direct-Anthropic CLI sends its requests; the key is never
        # checked. OpenRouter and Codex turns override both.
        monkeypatch.setenv("ANTHROPIC_BASE_URL", f"{provider.url}/anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-billing-e2e")
        if cli_max_retries is not None:
            monkeypatch.setenv("CLAUDE_CODE_MAX_RETRIES", str(cli_max_retries))
        monkeypatch.setattr(sdk_env, "config", config)
        monkeypatch.setattr(
            openrouter_cost,
            "_GENERATION_URL",
            f"{provider.url}/openrouter/api/v1/generation",
        )

    async def __call__(
        self, session: ChatSession, prompt: str, *, new_pod: bool
    ) -> None:
        self._turn += 1
        if new_pod:
            # Only what the turn uploaded survives a move to another pod.
            shutil.rmtree(self._config_dir / "projects", ignore_errors=True)
        with ExitStack() as stack:
            for name, value in self._stubs(session).items():
                stack.enter_context(patch.object(sdk_service, name, value))
            async for _ in sdk_service.stream_chat_completion_sdk(
                session_id=session.session_id,
                message=prompt,
                is_user_message=True,
                user_id=session.user_id,
                session=session,
                envelope=root_envelope(
                    str(uuid.uuid4()), session_id=session.session_id
                ),
                credential_lease=self._lease,
            ):
                pass

    def _stubs(self, session: ChatSession) -> dict[str, Any]:
        return {
            "config": self._config,
            "ClaudeSDKClient": MagicMock(side_effect=self._client),
            "AsyncClusterLock": MagicMock(side_effect=_lock),
            "_make_sdk_cwd": MagicMock(return_value=str(self._cwd)),
            "get_chat_session": AsyncMock(return_value=session),
            "upsert_chat_session": AsyncMock(return_value=session),
            "download_transcript": AsyncMock(side_effect=self._download),
            "upload_transcript": AsyncMock(side_effect=self._upload),
            "compact_transcript": AsyncMock(return_value=None),
            "get_redis_async": AsyncMock(),
            "propagate_attributes": MagicMock(),
            "_build_system_prompt": AsyncMock(
                return_value=("You are a test assistant. Reply briefly.", None)
            ),
            "build_skills_context": AsyncMock(return_value=""),
            "build_skills_update_notice": AsyncMock(return_value=None),
            "build_session_context": AsyncMock(return_value=""),
            "inject_user_context": AsyncMock(return_value=None),
            "create_copilot_mcp_server": MagicMock(),
            "create_security_hooks": MagicMock(),
            "get_copilot_tool_names": MagicMock(return_value=[]),
            "get_sdk_disallowed_tools": MagicMock(return_value=[]),
            "get_user_tier": AsyncMock(return_value=None),
            "_resolve_dynamic_max_budget_usd": AsyncMock(return_value=100.0),
            "build_turn_budget_block": AsyncMock(return_value=""),
            "drain_pending_safe": AsyncMock(return_value=[]),
            "drain_pending_for_persist": AsyncMock(return_value=[]),
            "resolve_answered": AsyncMock(return_value=[]),
            "clear_pending_question": AsyncMock(),
        }

    def _client(
        self, *args: Any, options: ClaudeAgentOptions, **kwargs: Any
    ) -> ClaudeSDKClient:
        self.resumed.append((self._turn, options.resume))
        # The copilot's own tools and hooks are not what is under test; the
        # CLI answers the provider's tool call with "no such tool" instead.
        options.mcp_servers = {}
        options.hooks = None
        options.allowed_tools = []
        options.disallowed_tools = []
        return ClaudeSDKClient(*args, options=options, **kwargs)

    async def _upload(
        self,
        *,
        session_id: str,
        content: bytes,
        message_count: int,
        mode: Any,
        **_: Any,
    ) -> None:
        self._transcripts[session_id] = TranscriptDownload(
            content=content, message_count=message_count, mode=mode
        )

    async def _download(
        self, user_id: str, session_id: str, *_: Any, **__: Any
    ) -> TranscriptDownload | None:
        return self._transcripts.get(session_id)


def _lock(*_: Any, owner_id: str = "", **__: Any) -> MagicMock:
    return MagicMock(
        try_acquire=AsyncMock(return_value=owner_id),
        refresh=AsyncMock(),
        release=AsyncMock(),
    )


class _CodexAccount(CodexAgentSession):
    """The production Codex session over the fake Responses endpoint.

    Only the model list is answered here: production fetches it through the
    platform's guarded HTTP client, which refuses loopback addresses.
    """

    async def models(self) -> list[CodexModelInfo]:
        return [
            CodexModelInfo(
                model=CODEX_MODEL,
                display_name="Codex",
                is_default=True,
                hidden=False,
                default_reasoning_effort="medium",
                supported_reasoning_efforts=["low", "medium", "high"],
            )
        ]


def _codex_lease(provider: FakeProvider) -> CodexCredentialLease:
    credentials = _codex_credentials()
    client = AsyncOpenAI(api_key="codex-billing-e2e", base_url=f"{provider.url}/codex")
    session = CodexHttpSession(
        credentials,
        turn_timeout_seconds=120,
        tool_timeout_seconds=60,
        client=client,
    )
    return CodexCredentialLease(credentials, _CodexAccount(session, credentials))


def _codex_credentials() -> OAuth2Credentials:
    # An unsigned JWT: the platform reads its claims, never its signature.
    token = "eyJhbGciOiJub25lIn0.eyJleHAiOjk5OTk5OTk5OTksImVtYWlsIjoiYUBiLmMifQ.sig"
    bundle = CodexAuthBundleV1(
        tokens=CodexAuthTokensV1(
            id_token=SecretStr(token),
            access_token=SecretStr(token),
            refresh_token=SecretStr("refresh"),
        ),
        codex_runtime_version="http",
    )
    return credentials_from_bundle(bundle).model_copy(
        update={"id": CODEX_CREDENTIAL_ID}
    )
