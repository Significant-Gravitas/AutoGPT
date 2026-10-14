"""The orchestrator block's SDK launch carries the user's key and nothing else.

A host may hold a Claude login (``CLAUDE_CODE_OAUTH_TOKEN``, a deployment
``ANTHROPIC_API_KEY``, a base URL) and every other backend secret in its own
environment, which the SDK merges under ``options.env``.  The launch must
override every one of them.
"""

import os
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock, patch

import claude_agent_sdk
import pytest
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage
from pydantic import SecretStr

from backend.blocks.orchestrator import OrchestratorBlock
from backend.data.model import APIKeyCredentials

# Other routes' keys, integration keys and service credentials, all fake.
_PARENT_SECRETS = {
    "CHAT_OPENROUTER_API_KEY": "sk-or-FAKE-chat-route",
    "CHAT_LOCAL_API_KEY": "FAKE-local-server-key",
    "OPEN_ROUTER_API_KEY": "sk-or-FAKE-integration",
    "E2B_API_KEY": "e2b_FAKE",
    "DATABASE_URL": "postgresql://app:FAKE-db-password@db:5432/app",
    "SUPABASE_SERVICE_ROLE_KEY": "FAKE-service-role",
    "LANGFUSE_SECRET_KEY": "sk-lf-FAKE",
    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer FAKE-otel",
}
_PARENT_ENV = {
    "ANTHROPIC_BASE_URL": "https://remote-gateway.example.com",
    "ANTHROPIC_API_KEY": "sk-ant-deployment-key",
    "ANTHROPIC_AUTH_TOKEN": "inherited-token",
    "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-inherited",
    **_PARENT_SECRETS,
}


class _RecordingClient:
    """Stands in for ``ClaudeSDKClient``; records the options it was given."""

    launched: list[ClaudeAgentOptions] = []

    def __init__(self, options: ClaudeAgentOptions) -> None:
        self.launched.append(options)

    async def __aenter__(self) -> "_RecordingClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def query(self, prompt: str) -> None:
        return None

    async def receive_response(self) -> AsyncGenerator[Any, None]:
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sess-1",
        )


async def _launch_env(provider: str, key: str) -> dict[str, str]:
    """What the CLI process would see: the parent env under ``options.env``."""
    _RecordingClient.launched = []
    input_data = MagicMock(sys_prompt="", prompt="hi")
    input_data.model.value = "claude-test"
    credentials = APIKeyCredentials(
        id="cred-1",
        provider=provider,
        api_key=SecretStr(key),
        title="user key",
        expires_at=None,
    )
    with (
        patch.object(claude_agent_sdk, "ClaudeSDKClient", _RecordingClient),
        patch.dict("os.environ", _PARENT_ENV),
    ):
        parent = dict(os.environ)
        async for _ in OrchestratorBlock()._execute_tools_sdk_mode(
            input_data=input_data,
            credentials=credentials,
            tool_functions=[],
            prompt=[{"role": "user", "content": "hi"}],
            execution_params=MagicMock(graph_exec_id="graph-exec-1"),
            execution_processor=MagicMock(),
        ):
            pass
    [options] = _RecordingClient.launched
    return {**parent, **options.env}


@pytest.mark.asyncio
async def test_anthropic_key_goes_direct_without_oauth():
    env = await _launch_env("anthropic", "sk-ant-user-key")
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-user-key"
    assert env["ANTHROPIC_AUTH_TOKEN"] == ""
    assert env["ANTHROPIC_BASE_URL"] == ""
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""


@pytest.mark.asyncio
async def test_openrouter_key_goes_to_openrouter_without_oauth():
    env = await _launch_env("open_router", "sk-or-user-key")
    # The chat's OpenRouter credential shape: the key as the bearer, the
    # deployment's inherited key blanked.
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-or-user-key"
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider, key",
    [("anthropic", "sk-ant-user-key"), ("open_router", "sk-or-user-key")],
)
async def test_the_launch_keeps_no_other_secret_of_the_parent(provider, key):
    env = await _launch_env(provider, key)
    for name, secret in _PARENT_SECRETS.items():
        assert not any(secret in value for value in env.values()), name
