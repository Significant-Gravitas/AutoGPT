"""Langfuse tracing integration for Claude Agent SDK.

This module provides modular, non-invasive observability for SDK sessions.
All tracing is opt-in (only active when Langfuse credentials are configured)
and designed to not affect the core execution flow.

Usage:
    async with TracedSession(session_id, user_id) as tracer:
        # Your SDK code here
        tracer.log_user_message(message)
        async for sdk_msg in client.receive_messages():
            tracer.log_sdk_message(sdk_msg)
        tracer.log_result(result_message)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from backend.util.settings import Settings

if TYPE_CHECKING:
    from claude_agent_sdk import Message, ResultMessage

logger = logging.getLogger(__name__)
settings = Settings()


def _is_langfuse_configured() -> bool:
    """Check if Langfuse credentials are configured."""
    return bool(
        settings.secrets.langfuse_public_key and settings.secrets.langfuse_secret_key
    )


@dataclass
class ToolSpan:
    """Tracks a single tool call for tracing."""

    tool_call_id: str
    tool_name: str
    input: dict[str, Any]
    start_time: float = field(default_factory=time.perf_counter)
    output: str | None = None
    success: bool = True
    end_time: float | None = None


@dataclass
class GenerationSpan:
    """Tracks an LLM generation (text output) for tracing."""

    text: str = ""
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    tool_calls: list[ToolSpan] = field(default_factory=list)


class TracedSession:
    """Context manager for tracing a Claude Agent SDK session with Langfuse.

    Automatically creates a trace with:
    - Session-level metadata (user_id, session_id)
    - Generation spans for LLM outputs
    - Tool call spans with input/output
    - Token usage and cost (from ResultMessage)

    If Langfuse is not configured, all methods are no-ops.
    """

    def __init__(
        self,
        session_id: str,
        user_id: str | None = None,
        system_prompt: str | None = None,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.system_prompt = system_prompt
        self.enabled = _is_langfuse_configured()

        # Internal state
        self._trace: Any = None
        self._langfuse: Any = None
        self._user_message: str | None = None
        self._generations: list[GenerationSpan] = []
        self._current_generation: GenerationSpan | None = None
        self._pending_tools: dict[str, ToolSpan] = {}
        self._start_time: float = 0

    async def __aenter__(self) -> TracedSession:
        """Start the trace."""
        if not self.enabled:
            return self

        try:
            from langfuse import get_client

            self._langfuse = get_client()
            self._start_time = time.perf_counter()

            # Create the root trace
            self._trace = self._langfuse.trace(
                name="copilot-sdk-session",
                session_id=self.session_id,
                user_id=self.user_id,
                metadata={
                    "sdk": "claude-agent-sdk",
                    "has_system_prompt": bool(self.system_prompt),
                },
            )
            logger.debug(f"[Tracing] Started trace for session {self.session_id}")

        except Exception as e:
            logger.warning(f"[Tracing] Failed to start trace: {e}")
            self.enabled = False

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End the trace and flush to Langfuse."""
        if not self.enabled or not self._trace:
            return

        try:
            # Finalize any open generation
            self._finalize_current_generation()

            # Add generations as spans
            for gen in self._generations:
                self._trace.span(
                    name="llm-generation",
                    start_time=gen.start_time,
                    end_time=gen.end_time or time.perf_counter(),
                    output=gen.text[:1000] if gen.text else None,  # Truncate
                    metadata={"tool_calls": len(gen.tool_calls)},
                )

                # Add tool calls as nested spans
                for tool in gen.tool_calls:
                    self._trace.span(
                        name=f"tool:{tool.tool_name}",
                        start_time=tool.start_time,
                        end_time=tool.end_time or time.perf_counter(),
                        input=tool.input,
                        output=tool.output[:500] if tool.output else None,
                        metadata={
                            "tool_call_id": tool.tool_call_id,
                            "success": tool.success,
                        },
                    )

            # Update trace with final status
            status = "error" if exc_type else "success"
            self._trace.update(
                output=self._generations[-1].text[:500] if self._generations else None,
                metadata={"status": status, "num_generations": len(self._generations)},
            )

            # Flush asynchronously (Langfuse handles this in background)
            logger.debug(
                f"[Tracing] Completed trace for session {self.session_id}, "
                f"{len(self._generations)} generations"
            )

        except Exception as e:
            logger.warning(f"[Tracing] Failed to finalize trace: {e}")

    def log_user_message(self, message: str) -> None:
        """Log the user's input message."""
        if not self.enabled or not self._trace:
            return

        self._user_message = message
        try:
            self._trace.update(input=message[:1000])
        except Exception as e:
            logger.debug(f"[Tracing] Failed to log user message: {e}")

    def log_sdk_message(self, sdk_message: Message) -> None:
        """Log an SDK message (automatically categorizes by type)."""
        if not self.enabled:
            return

        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ResultMessage,
                TextBlock,
                ToolResultBlock,
                ToolUseBlock,
                UserMessage,
            )

            if isinstance(sdk_message, AssistantMessage):
                # Start a new generation if needed
                if self._current_generation is None:
                    self._current_generation = GenerationSpan()
                    self._generations.append(self._current_generation)

                for block in sdk_message.content:
                    if isinstance(block, TextBlock) and block.text:
                        self._current_generation.text += block.text

                    elif isinstance(block, ToolUseBlock):
                        tool_span = ToolSpan(
                            tool_call_id=block.id,
                            tool_name=block.name,
                            input=block.input or {},
                        )
                        self._pending_tools[block.id] = tool_span
                        if self._current_generation:
                            self._current_generation.tool_calls.append(tool_span)

            elif isinstance(sdk_message, UserMessage):
                # UserMessage carries tool results
                content = sdk_message.content
                blocks = content if isinstance(content, list) else []
                for block in blocks:
                    if isinstance(block, ToolResultBlock) and block.tool_use_id:
                        tool_span = self._pending_tools.get(block.tool_use_id)
                        if tool_span:
                            tool_span.end_time = time.perf_counter()
                            tool_span.success = not (block.is_error or False)
                            tool_span.output = self._extract_tool_output(block.content)

                # After tool results, finalize current generation
                # (SDK will start a new AssistantMessage for continuation)
                self._finalize_current_generation()

            elif isinstance(sdk_message, ResultMessage):
                self._log_result(sdk_message)

        except Exception as e:
            logger.debug(f"[Tracing] Failed to log SDK message: {e}")

    def _log_result(self, result: ResultMessage) -> None:
        """Log the final result with usage and cost."""
        if not self.enabled or not self._trace:
            return

        try:
            # Extract usage info
            usage = result.usage or {}
            metadata: dict[str, Any] = {
                "duration_ms": result.duration_ms,
                "duration_api_ms": result.duration_api_ms,
                "num_turns": result.num_turns,
                "is_error": result.is_error,
            }

            if result.total_cost_usd is not None:
                metadata["cost_usd"] = result.total_cost_usd

            if usage:
                metadata["usage"] = usage

            self._trace.update(metadata=metadata)

            # Log as a generation for proper Langfuse cost/usage tracking
            if usage or result.total_cost_usd:
                self._trace.generation(
                    name="claude-sdk-completion",
                    model="claude-sonnet-4-20250514",  # SDK default model
                    usage=(
                        {
                            "input": usage.get("input_tokens", 0),
                            "output": usage.get("output_tokens", 0),
                            "total": usage.get("input_tokens", 0)
                            + usage.get("output_tokens", 0),
                        }
                        if usage
                        else None
                    ),
                    metadata={"cost_usd": result.total_cost_usd},
                )

            logger.debug(
                f"[Tracing] Logged result: {result.num_turns} turns, "
                f"${result.total_cost_usd:.4f} cost"
                if result.total_cost_usd
                else f"[Tracing] Logged result: {result.num_turns} turns"
            )

        except Exception as e:
            logger.debug(f"[Tracing] Failed to log result: {e}")

    def _finalize_current_generation(self) -> None:
        """Mark the current generation as complete."""
        if self._current_generation:
            self._current_generation.end_time = time.perf_counter()
            self._current_generation = None

    @staticmethod
    def _extract_tool_output(content: str | list[dict[str, str]] | None) -> str:
        """Extract string output from tool result content."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                item.get("text", "") for item in content if item.get("type") == "text"
            ]
            return "".join(parts) if parts else str(content)
        return str(content) if content else ""


@asynccontextmanager
async def traced_session(
    session_id: str,
    user_id: str | None = None,
    system_prompt: str | None = None,
):
    """Convenience async context manager for tracing SDK sessions.

    Usage:
        async with traced_session(session_id, user_id) as tracer:
            tracer.log_user_message(message)
            async for msg in client.receive_messages():
                tracer.log_sdk_message(msg)
    """
    tracer = TracedSession(session_id, user_id, system_prompt)
    async with tracer:
        yield tracer


def create_tracing_hooks(tracer: TracedSession) -> dict[str, Any]:
    """Create SDK hooks for fine-grained Langfuse tracing.

    These hooks capture precise timing for tool executions and failures
    that may not be visible in the message stream.

    Designed to be merged with security hooks:
        hooks = {**security_hooks, **create_tracing_hooks(tracer)}

    Args:
        tracer: The active TracedSession instance

    Returns:
        Hooks configuration dict for ClaudeAgentOptions
    """
    if not tracer.enabled:
        return {}

    try:
        from claude_agent_sdk import HookMatcher
        from claude_agent_sdk.types import HookContext, HookInput, SyncHookJSONOutput

        async def trace_pre_tool_use(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Record tool start time for accurate duration tracking."""
            _ = context
            if not tool_use_id:
                return {}
            tool_name = str(input_data.get("tool_name", "unknown"))
            tool_input = input_data.get("tool_input", {})

            # Record start time in pending tools
            tracer._pending_tools[tool_use_id] = ToolSpan(
                tool_call_id=tool_use_id,
                tool_name=tool_name,
                input=tool_input if isinstance(tool_input, dict) else {},
            )
            return {}

        async def trace_post_tool_use(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Record tool completion for duration calculation."""
            _ = context
            if tool_use_id and tool_use_id in tracer._pending_tools:
                tracer._pending_tools[tool_use_id].end_time = time.perf_counter()
                tracer._pending_tools[tool_use_id].success = True
            return {}

        async def trace_post_tool_failure(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Record tool failures for error tracking."""
            _ = context
            if tool_use_id and tool_use_id in tracer._pending_tools:
                tracer._pending_tools[tool_use_id].end_time = time.perf_counter()
                tracer._pending_tools[tool_use_id].success = False
                error = input_data.get("error", "Unknown error")
                tracer._pending_tools[tool_use_id].output = f"ERROR: {error}"
            return {}

        return {
            "PreToolUse": [HookMatcher(matcher="*", hooks=[trace_pre_tool_use])],
            "PostToolUse": [HookMatcher(matcher="*", hooks=[trace_post_tool_use])],
            "PostToolUseFailure": [
                HookMatcher(matcher="*", hooks=[trace_post_tool_failure])
            ],
        }

    except ImportError:
        logger.debug("[Tracing] SDK not available for hook-based tracing")
        return {}


def merge_hooks(*hook_dicts: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple hook configurations into one.

    Combines hook matchers for the same event type, allowing both
    security and tracing hooks to coexist.

    Usage:
        combined = merge_hooks(security_hooks, tracing_hooks)
    """
    result: dict[str, list[Any]] = {}
    for hook_dict in hook_dicts:
        for event_name, matchers in hook_dict.items():
            if event_name not in result:
                result[event_name] = []
            result[event_name].extend(matchers)
    return result
