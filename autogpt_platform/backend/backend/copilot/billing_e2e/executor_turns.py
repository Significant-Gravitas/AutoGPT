"""Turns handed to the copilot executor, the way its queue hands them over.

``CoPilotProcessor._execute_async`` runs each turn as in production: it
snapshots which trial the turn is spent from, starts the engine, wraps its
stream in the heartbeat and drains it. Stubbed: the session store, the Redis
stream the browser reads, and the engine flag, since each scenario names its
engine. The engines run on the billing suite's own harnesses, except that
the budget lookups a trial changes run for real.
"""

import logging
import shutil
import threading
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from backend.copilot import budget_gate
from backend.copilot import model as chat_model
from backend.copilot import rate_limit, stream_registry, tools
from backend.copilot.baseline import service as baseline
from backend.copilot.config import ChatConfig
from backend.copilot.executor import processor
from backend.copilot.executor.processor import CoPilotProcessor
from backend.copilot.executor.utils import CoPilotExecutionEntry, CoPilotLogMetadata
from backend.copilot.model import ChatSession
from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk import tool_adapter
from backend.copilot.sdk.tool_adapter import MCP_TOOL_PREFIX
from backend.copilot.tools.base import BaseTool
from backend.copilot.tree import root_envelope

from .baseline_turns import BaselineTurns
from .provider import FakeProvider
from .sdk_turns import SdkTurns

logger = logging.getLogger(__name__)


class Executor:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ChatConfig,
        session: ChatSession,
        *,
        use_sdk: bool,
    ) -> None:
        # The error each turn ended with, None for a turn that completed.
        self.errors: list[str | None] = []
        self._processor = CoPilotProcessor()
        monkeypatch.setattr(
            chat_model, "get_chat_session", AsyncMock(return_value=session)
        )
        # Where a refused turn saves its not-started marker.
        monkeypatch.setattr(
            budget_gate, "upsert_chat_session", AsyncMock(return_value=session)
        )
        monkeypatch.setattr(processor, "ChatConfig", MagicMock(return_value=config))
        monkeypatch.setattr(
            processor, "resolve_use_sdk", AsyncMock(return_value=use_sdk)
        )
        monkeypatch.setattr(
            processor, "_building_mode_forces_sdk", AsyncMock(return_value=False)
        )
        monkeypatch.setattr(
            stream_registry, "stream_and_publish", lambda **kwargs: kwargs["stream"]
        )
        monkeypatch.setattr(stream_registry, "publish_chunk", AsyncMock())
        monkeypatch.setattr(
            stream_registry,
            "mark_session_completed",
            AsyncMock(side_effect=self._completed),
        )

    async def run(self, session: ChatSession, prompt: str) -> None:
        entry = CoPilotExecutionEntry(
            session_id=session.session_id,
            turn_id=str(uuid.uuid4()),
            user_id=session.user_id,
            message=prompt,
            envelope=root_envelope(str(uuid.uuid4()), session_id=session.session_id),
        )
        await self._processor._execute_async(
            entry, threading.Event(), MagicMock(), CoPilotLogMetadata(logger=logger)
        )

    async def _completed(
        self, _session_id: str, *, error_message: str | None = None, **_: Any
    ) -> None:
        self.errors.append(error_message)


class SdkTurnsViaExecutor(SdkTurns):
    """SDK turns from the executor, with the trial sizing each turn's budget.

    With ``served_tools``, the CLI runs the model's tool calls on the
    copilot's real MCP server and tool adapter, serving only those tools.
    """

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        provider: FakeProvider,
        config: ChatConfig,
        executor: Executor,
        *,
        served_tools: dict[str, BaseTool] | None = None,
    ) -> None:
        super().__init__(tmp_path, monkeypatch, provider, config)
        self._executor = executor
        self._tools = served_tools or {}
        if self._tools:
            monkeypatch.setattr(tool_adapter, "TOOL_REGISTRY", self._tools)
        # The CLI's spending cap for each launch, by turn.
        self.budgets: list[tuple[int, float | None]] = []

    async def __call__(
        self, session: ChatSession, prompt: str, *, new_pod: bool = False
    ) -> None:
        self._turn += 1
        if new_pod:
            shutil.rmtree(self._config_dir / "projects", ignore_errors=True)
        with ExitStack() as stack:
            for name, value in self._stubs(session).items():
                stack.enter_context(patch.object(sdk_service, name, value))
            await self._executor.run(session, prompt)

    def _stubs(self, session: ChatSession) -> dict[str, Any]:
        stubs = super()._stubs(session)
        del stubs["get_user_tier"]
        del stubs["resolve_turn_budget"]
        if self._tools:
            del stubs["create_copilot_mcp_server"]
        return stubs

    def _client(
        self, *args: Any, options: ClaudeAgentOptions, **kwargs: Any
    ) -> ClaudeSDKClient:
        self.budgets.append((self._turn, options.max_budget_usd))
        if not self._tools:
            return super()._client(*args, options=options, **kwargs)
        self.resumed.append((self._turn, options.resume))
        options.hooks = None
        options.allowed_tools = [f"{MCP_TOOL_PREFIX}{name}" for name in self._tools]
        options.disallowed_tools = []
        return ClaudeSDKClient(*args, options=options, **kwargs)


class BaselineTurnsViaExecutor(BaselineTurns):
    """Baseline turns from the executor, with the trial sizing the budget the
    model is told about."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ChatConfig,
        executor: Executor,
        *,
        base_url: str,
        model: str,
    ) -> None:
        super().__init__(monkeypatch, config, base_url=base_url, model=model)
        self._executor = executor
        self.user_context = AsyncMock(return_value=None)
        # BaselineTurns stubs it; the trial is what sizes this budget.
        monkeypatch.setattr(baseline, "build_budget_ctx", rate_limit.build_budget_ctx)
        monkeypatch.setattr(baseline, "inject_user_context", self.user_context)

    async def __call__(
        self, session: ChatSession, prompt: str, *, new_pod: bool = False
    ) -> None:
        await self._executor.run(session, prompt)

    def run_registry_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run the model's tool calls through the copilot's tool registry."""
        monkeypatch.setattr(baseline, "execute_tool", tools.execute_tool)
