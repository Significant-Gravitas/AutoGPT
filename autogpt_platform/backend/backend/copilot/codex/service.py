from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

from openai.types.chat import ChatCompletionToolParam
from openai_codex.generated.v2_all import (
    AgentMessageDeltaNotification,
    ReasoningSummaryTextDeltaNotification,
)
from openai_codex.models import Notification

from backend.copilot.config import CopilotLlmModel, CopilotMode
from backend.copilot.context import set_execution_context
from backend.copilot.expert_context import build_expert_identity_suffix
from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    maybe_append_user_message,
    upsert_chat_session,
)
from backend.copilot.pending_message_helpers import (
    drain_pending_safe,
    persist_pending_as_user_rows,
)
from backend.copilot.permissions import CopilotPermissions, all_known_tool_names
from backend.copilot.prompting import SHARED_TOOL_NOTES, get_graphiti_supplement
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamFinish,
    StreamFinishStep,
    StreamReasoningDelta,
    StreamReasoningEnd,
    StreamReasoningStart,
    StreamStart,
    StreamStartStep,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
    StreamUsage,
)
from backend.copilot.service import CACHEABLE_SYSTEM_PROMPT, strip_user_context_tags
from backend.copilot.token_tracking import persist_and_record_usage
from backend.copilot.tools import ToolGroup, execute_tool, get_available_tools
from backend.integrations.codex.models import (
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
)
from backend.integrations.codex.transport import CodexTransport, get_codex_transport
from backend.integrations.credential_lease import CredentialLease
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)
_UNSUPPORTED_CODEX_TOOLS = {
    "enter_agent_building_mode",
    "run_sub_session",
}


async def stream_chat_completion_codex(
    session_id: str,
    message: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    session: ChatSession | None = None,
    file_ids: list[str] | None = None,
    permissions: CopilotPermissions | None = None,
    context: dict[str, str] | None = None,
    mode: CopilotMode | None = None,
    model: CopilotLlmModel | None = None,
    request_arrival_at: float = 0.0,
    organization_id: str | None = None,
    team_id: str | None = None,
    *,
    credential_lease: CredentialLease,
    transport: CodexTransport | None = None,
    **_kwargs: Any,
) -> AsyncGenerator[StreamBaseResponse, None]:
    del mode, request_arrival_at
    if user_id is None:
        raise RuntimeError("codex_user_required")
    if file_ids:
        raise RuntimeError("codex_file_attachments_unsupported")
    if session is None:
        session = await get_chat_session(session_id, user_id)
    if session is None:
        raise NotFoundError(f"Session {session_id} not found")
    if session.user_id != user_id:
        raise RuntimeError("codex_session_route_mismatch")
    if session.metadata.llm_auth_provider != "codex":
        raise RuntimeError("codex_session_route_mismatch")
    if session.metadata.llm_credential_id != credential_lease.credentials.id:
        raise RuntimeError("codex_session_route_mismatch")
    active_session: ChatSession = session

    if active_session.organization_id is None and organization_id:
        active_session.organization_id = organization_id
        active_session.team_id = team_id

    session_message_was_sanitized = False
    if message and is_user_message:
        sanitized_message = strip_user_context_tags(message)
        if sanitized_message != message:
            for existing in reversed(active_session.messages):
                if existing.role == "user" and existing.content == message:
                    existing.content = sanitized_message
                    session_message_was_sanitized = True
                    break
        message = sanitized_message

    message_was_appended = maybe_append_user_message(
        active_session,
        message,
        is_user_message,
    )
    if message_was_appended or session_message_was_sanitized:
        if is_user_message and not active_session.title and message:
            active_session.title = _fallback_title(message)
        active_session = await upsert_chat_session(active_session)

    pending = await drain_pending_safe(session_id, "[Codex]")
    if pending:
        await persist_pending_as_user_rows(
            active_session,
            None,
            pending,
            log_prefix="[Codex]",
            content_of=lambda pending_message: strip_user_context_tags(
                pending_message.content
            ),
        )

    graphiti_enabled = await is_enabled_for_user(user_id)
    disabled_groups: list[ToolGroup] = [] if graphiti_enabled else ["graphiti"]
    tools = get_available_tools(disabled_groups=disabled_groups)
    tools = [
        tool
        for tool in tools
        if tool.get("function", {}).get("name") not in _UNSUPPORTED_CODEX_TOOLS  # type: ignore[union-attr]
    ]
    if permissions is not None:
        tools = _filter_tools_by_permissions(tools, permissions)
    dynamic_tools = _to_dynamic_tools(tools)
    allowed_tool_names = {tool.name for tool in dynamic_tools}

    set_execution_context(user_id, active_session, permissions=permissions)
    prompt = _render_transcript(active_session.messages, context)
    instructions = CACHEABLE_SYSTEM_PROMPT + SHARED_TOOL_NOTES
    if graphiti_enabled:
        instructions += get_graphiti_supplement()
    instructions += await build_expert_identity_suffix(
        active_session.user_id,
        active_session.expert_id,
    )
    instructions += (
        "\nUse only the AutoGPT dynamic tools exposed for this turn. "
        "Do not request shell, filesystem, network, or approval tools from "
        "the Codex runtime itself."
    )

    event_queue: asyncio.Queue[StreamBaseResponse] = asyncio.Queue()
    text_id = str(uuid.uuid4())
    reasoning_id = str(uuid.uuid4())
    text_started = False
    reasoning_started = False
    streamed_text_parts: list[str] = []
    tool_execution_lock = asyncio.Lock()

    async def on_event(notification: Notification) -> None:
        nonlocal text_started, reasoning_started
        payload = notification.payload
        if isinstance(payload, AgentMessageDeltaNotification):
            streamed_text_parts.append(payload.delta)
            if not text_started:
                text_started = True
                await event_queue.put(StreamTextStart(id=text_id))
            await event_queue.put(StreamTextDelta(id=text_id, delta=payload.delta))
        elif isinstance(payload, ReasoningSummaryTextDeltaNotification):
            if not reasoning_started:
                reasoning_started = True
                await event_queue.put(StreamReasoningStart(id=reasoning_id))
            await event_queue.put(
                StreamReasoningDelta(id=reasoning_id, delta=payload.delta)
            )

    async def run_tool(call: CodexDynamicToolCall) -> CodexDynamicToolResult:
        nonlocal active_session
        if call.tool not in allowed_tool_names:
            return CodexDynamicToolResult(
                content="codex_tool_not_allowed",
                success=False,
            )
        if not isinstance(call.arguments, dict):
            return CodexDynamicToolResult(
                content="codex_tool_request_invalid",
                success=False,
            )
        arguments = cast(dict[str, Any], call.arguments)
        async with tool_execution_lock:
            set_execution_context(user_id, active_session, permissions=permissions)
            await event_queue.put(
                StreamToolInputStart(
                    toolCallId=call.call_id,
                    toolName=call.tool,
                )
            )
            await event_queue.put(
                StreamToolInputAvailable(
                    toolCallId=call.call_id,
                    toolName=call.tool,
                    input=arguments,
                )
            )
            active_session.announce_inflight_tool_call(call.tool, arguments)
            tool_call = {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.tool,
                    "arguments": json.dumps(arguments, default=str),
                },
            }
            try:
                result = await execute_tool(
                    tool_name=call.tool,
                    parameters=arguments,
                    user_id=user_id,
                    session=active_session,
                    tool_call_id=call.call_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[Codex] Dynamic tool %s failed", call.tool)
                result = StreamToolOutputAvailable(
                    toolCallId=call.call_id,
                    toolName=call.tool,
                    output="codex_tool_execution_failed",
                    success=False,
                )
            await event_queue.put(result)
            content = (
                result.output
                if isinstance(result.output, str)
                else json.dumps(result.output, default=str)
            )
            active_session.messages.extend(
                [
                    ChatMessage(role="assistant", tool_calls=[tool_call]),
                    ChatMessage(
                        role="tool",
                        name=call.tool,
                        tool_call_id=call.call_id,
                        content=content,
                    ),
                ]
            )
            active_session = await upsert_chat_session(active_session)
            set_execution_context(user_id, active_session, permissions=permissions)
            return CodexDynamicToolResult(content=content, success=result.success)

    message_id = str(uuid.uuid4())
    yield StreamStart(messageId=message_id, sessionId=session_id)
    yield StreamStartStep()

    active_transport = transport or get_codex_transport()
    invocation = asyncio.create_task(
        active_transport.invoke_agent(
            credential_lease,
            CodexInvocationRequest(
                prompt=prompt,
                instructions=instructions,
                effort="high" if model == "advanced" else "medium",
            ),
            dynamic_tools,
            run_tool,
            on_event,
        )
    )
    try:
        while not invocation.done():
            queued = asyncio.create_task(event_queue.get())
            done, _ = await asyncio.wait(
                {invocation, queued},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if queued in done:
                yield queued.result()
                continue
            queued.cancel()
            await asyncio.gather(queued, return_exceptions=True)
        while not event_queue.empty():
            yield event_queue.get_nowait()
        result = await invocation
    except (asyncio.CancelledError, GeneratorExit):
        invocation.cancel()
        await asyncio.gather(invocation, return_exceptions=True)
        raise
    except Exception:
        invocation.cancel()
        await asyncio.gather(invocation, return_exceptions=True)
        if reasoning_started:
            yield StreamReasoningEnd(id=reasoning_id)
        if text_started:
            yield StreamTextEnd(id=text_id)
        yield StreamFinishStep()
        raise
    finally:
        active_session.clear_inflight_tool_calls()

    if not text_started and result.final_response:
        text_started = True
        streamed_text_parts.append(result.final_response)
        yield StreamTextStart(id=text_id)
        yield StreamTextDelta(id=text_id, delta=result.final_response)
    if reasoning_started:
        yield StreamReasoningEnd(id=reasoning_id)
    if text_started:
        yield StreamTextEnd(id=text_id)

    displayed_response = "".join(streamed_text_parts) or result.final_response
    active_session.messages.append(
        ChatMessage(
            role="assistant",
            content=displayed_response,
            duration_ms=result.duration_ms,
        )
    )
    usage = result.usage
    if usage is not None:
        uncached_input = max(0, usage.input_tokens - usage.cached_input_tokens)
        await persist_and_record_usage(
            session=active_session,
            user_id=user_id,
            prompt_tokens=uncached_input,
            completion_tokens=usage.output_tokens,
            cache_read_tokens=usage.cached_input_tokens,
            log_prefix="[Codex]",
            cost_usd=None,
            model="codex-default",
            provider="codex",
            credential_id_override=credential_lease.credentials.id,
            extra_metadata={"billing_mode": "user_subscription"},
            execution_path="codex_native",
        )
    await upsert_chat_session(active_session)

    yield StreamFinishStep()
    if usage is not None:
        uncached_input = max(0, usage.input_tokens - usage.cached_input_tokens)
        yield StreamUsage(
            prompt_tokens=uncached_input,
            completion_tokens=usage.output_tokens,
            total_tokens=uncached_input + usage.output_tokens,
            cache_read_tokens=usage.cached_input_tokens,
        )
    yield StreamFinish()


def _to_dynamic_tools(
    tools: list[ChatCompletionToolParam],
) -> list[CodexDynamicToolSpec]:
    dynamic_tools = []
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name")
        if not isinstance(name, str):
            continue
        description = function.get("description")
        parameters = function.get("parameters")
        dynamic_tools.append(
            CodexDynamicToolSpec(
                name=name,
                description=(description if isinstance(description, str) else ""),
                input_schema=(
                    cast(dict[str, object], parameters)
                    if isinstance(parameters, dict)
                    else {"type": "object", "properties": {}}
                ),
            )
        )
    return dynamic_tools


def _filter_tools_by_permissions(
    tools: list[ChatCompletionToolParam],
    permissions: CopilotPermissions,
) -> list[ChatCompletionToolParam]:
    if permissions.is_empty():
        return tools
    allowed = permissions.effective_allowed_tools(all_known_tool_names())
    return [
        tool
        for tool in tools
        if tool.get("function", {}).get("name") in allowed  # type: ignore[union-attr]
    ]


def _render_transcript(
    messages: list[ChatMessage],
    context: dict[str, str] | None,
) -> str:
    rendered = ["Continue this AutoPilot conversation:"]
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            rendered.append(
                "ASSISTANT TOOL CALLS: " + json.dumps(message.tool_calls, default=str)
            )
        if message.content:
            role = message.role.upper()
            suffix = (
                f" [{message.name}; call_id={message.tool_call_id}]"
                if message.role == "tool"
                else ""
            )
            rendered.append(f"{role}{suffix}: {message.content}")
    if context:
        rendered.append("CURRENT PAGE CONTEXT: " + json.dumps(context, default=str))
    rendered.append("Respond to the latest user message and complete the task.")
    return "\n\n".join(rendered)


def _fallback_title(message: str) -> str:
    title = " ".join(message.strip().split()[:6])
    if len(title) <= 50:
        return title
    return title[:47].rstrip() + "..."
