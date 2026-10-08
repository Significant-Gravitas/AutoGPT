"""Everything a pai turn sets up before its first event, in the baseline's order.

Session housekeeping (orphan pruning, tag stripping, the user row, the
turn-start pending drain), model routing, the E2B box, the system prompt and
history load (concurrently), the title, the feature flags behind the prompt
supplements, first-turn context injection, the execution context, attachments
and the tool surface. Each step calls the same function the baseline calls.
"""

import asyncio
import logging
import tempfile
import uuid
from typing import Any

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, ConfigDict
from pydantic_ai.messages import ModelMessage, UserContent

from backend.blocks.desktop._common import workspace_volume_mounts
from backend.copilot.baseline.service import (
    _background_tasks,
    _fetch_graphiti_context,
    _filter_tools_by_permissions,
    _prepare_baseline_attachments,
)
from backend.copilot.builder_context import build_builder_system_prompt_suffix
from backend.copilot.config import ChatConfig, CopilotLLMModel
from backend.copilot.context import set_execution_context
from backend.copilot.expert_context import build_expert_identity_suffix
from backend.copilot.expert_kickoff import is_expert_kickoff_turn
from backend.copilot.gate import active_mode
from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.model import (
    ChatSession,
    clear_pending_question,
    get_chat_session,
    maybe_append_user_message,
)
from backend.copilot.pending_message_helpers import (
    drain_pending_safe,
    drained_rows_entry,
    persist_pending_as_user_rows,
)
from backend.copilot.pending_messages import (
    PendingMessage,
    format_pending_as_user_message,
)
from backend.copilot.permissions import CopilotPermissions, denied_tool_names
from backend.copilot.response_model import StreamBaseResponse
from backend.copilot.service import (
    _build_system_prompt,
    _update_title_async,
    strip_user_context_tags,
)
from backend.copilot.session_cleanup import prune_orphan_tool_calls
from backend.copilot.tools import (
    ToolGroup,
    expert_tool_disabled_groups,
    get_available_tools,
    kickoff_turn_disabled_tools,
    origin_disabled_tools,
    tool_names_in_groups,
)
from backend.copilot.tools.e2b_sandbox import get_or_create_sandbox
from backend.copilot.tracking import track_user_message
from backend.copilot.transcript import (
    TranscriptDownload,
    detect_gap,
    extract_context_messages,
)
from backend.copilot.tree import TurnEnvelope
from backend.util.exceptions import NotFoundError
from backend.util.feature_flag import Flag, is_feature_enabled

from .history import HeldToolCall, chat_rows_to_messages, close_pending_calls
from .history_store import LoadedHistory, download_history
from .model import PaiRoute, resolve_route
from .prompt import PromptInputs, build_static_instructions, build_turn_context
from .turn_context import (
    build_user_prompt,
    first_turn_prefix,
    turn_context_blocks,
    url_context_hint,
)

logger = logging.getLogger(__name__)


class TurnRequest(BaseModel):
    """The processor's arguments, as the baseline receives them."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    message: str | None = None
    is_user_message: bool = True
    user_id: str | None = None
    session: ChatSession | None = None
    file_ids: list[str] | None = None
    permissions: CopilotPermissions | None = None
    envelope: TurnEnvelope | None = None
    context: dict[str, str] | None = None
    model: CopilotLLMModel | None = None
    request_arrival_at: float = 0.0
    organization_id: str | None = None
    team_id: str | None = None
    message_metadata: dict[str, Any] | None = None


class PreparedTurn(BaseModel):
    """What the turn runs with."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: ChatSession
    route: PaiRoute
    sandbox: Any = None
    static_instructions: str
    turn_context: str
    user_prompt: str | list[UserContent] | None
    message: str | None
    graphiti_enabled: bool
    tools: list[ChatCompletionToolParam]
    disabled_groups: list[ToolGroup]
    disabled_tools: frozenset[str]
    working_dir: str | None
    history: list[ModelMessage]
    held: list[HeldToolCall]
    upload_safe: bool
    opening_entries: list[StreamBaseResponse]
    turn_start: int
    message_id: str


class _Loaded(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    history: list[ModelMessage]
    held: list[HeldToolCall]
    upload_safe: bool


async def prepare_turn(request: TurnRequest, config: ChatConfig) -> PreparedTurn:
    session, message = await _open_session(request)
    pre_drain_count = len(session.messages)
    queued = await drain_pending_safe(request.session_id, "[PAI]")
    route = await resolve_route(request.model, request.user_id, config)
    sandbox = await _open_sandbox(session, request, config)
    first_turn = pre_drain_count <= 1
    inject = first_turn and request.is_user_message
    (base_prompt, understanding), loaded = await asyncio.gather(
        _build_system_prompt(request.user_id if inject else None),
        _load_history(request.user_id, session),
    )
    _maybe_title(session, message, request)
    graphiti = await is_enabled_for_user(request.user_id)
    experts = bool(request.user_id) and await is_feature_enabled(
        Flag.HIRE_EXPERTS, request.user_id or "", default=False
    )
    static = await _static_instructions(
        base_prompt, session, request, graphiti, experts
    )
    warm_ctx = (
        await _fetch_graphiti_context(request.user_id, session, message)
        if graphiti and request.user_id and first_turn
        else None
    )
    prompt_text = message
    if inject:
        prefixed = await first_turn_prefix(
            understanding, message or "", session, request.user_id, config
        )
        prompt_text = prefixed if prefixed is not None else message
    opening: list[StreamBaseResponse] = []
    if queued and await persist_pending_as_user_rows(
        session, None, queued, log_prefix="[PAI]"
    ):
        opening.append(drained_rows_entry(queued))
    else:
        queued = []
    blocks = await turn_context_blocks(
        session,
        user_id=request.user_id,
        envelope=request.envelope,
        is_user_message=request.is_user_message,
        warm_ctx=warm_ctx,
    )
    groups, disabled = _disabled(session, graphiti, experts)
    working_dir = (
        tempfile.mkdtemp(prefix=f"copilot-pai-{request.session_id[:8]}-")
        if request.file_ids and request.user_id
        else None
    )
    _set_context(request, session, sandbox, working_dir, groups, disabled)
    hint, images = await _attachments(request, working_dir)
    tools = get_available_tools(disabled_groups=groups, disabled_tools=disabled)
    if request.permissions is not None:
        tools = _filter_tools_by_permissions(tools, request.permissions)
    return PreparedTurn(
        session=session,
        route=route,
        sandbox=sandbox,
        static_instructions=static,
        turn_context=build_turn_context(blocks),
        user_prompt=build_user_prompt(
            prompt_text,
            hint=hint + url_context_hint(request.context),
            queued=queued,
            late_results=[],
            image_blocks=images,
        ),
        message=message,
        graphiti_enabled=graphiti,
        tools=tools,
        disabled_groups=groups,
        disabled_tools=disabled,
        working_dir=working_dir,
        history=loaded.history,
        held=loaded.held,
        upload_safe=loaded.upload_safe,
        opening_entries=opening,
        turn_start=pre_drain_count,
        message_id=str(uuid.uuid4()),
    )


async def _open_session(request: TurnRequest) -> tuple[ChatSession, str | None]:
    session = request.session or await get_chat_session(
        request.session_id, request.user_id
    )
    if not session:
        raise NotFoundError(
            f"Session {request.session_id} not found. Please create a new session first."
        )
    if session.organization_id is None and request.organization_id:
        session.organization_id = request.organization_id
        session.team_id = request.team_id
    prune_orphan_tool_calls(
        session.messages, log_prefix=f"[PAI] [{request.session_id[:12]}]"
    )
    message = strip_user_context_tags(request.message) if request.message else None
    if request.is_user_message and message and message.strip():
        await clear_pending_question(session)
    appended = maybe_append_user_message(
        session, message, request.is_user_message, request.message_metadata
    )
    if appended and request.is_user_message:
        track_user_message(
            user_id=request.user_id,
            session_id=request.session_id,
            message_length=len(message or ""),
            expert_id=session.expert_id,
            origin=session.metadata.origin,
            source_platform=session.metadata.source_platform,
        )
    return session, message


async def _open_sandbox(
    session: ChatSession, request: TurnRequest, config: ChatConfig
) -> Any:
    api_key = config.active_e2b_api_key
    if not api_key:
        return None
    try:
        return await get_or_create_sandbox(
            request.session_id,
            api_key=api_key,
            template=config.e2b_sandbox_template,
            timeout=config.e2b_sandbox_timeout,
            on_timeout=config.e2b_sandbox_on_timeout,
            volume_mounts=workspace_volume_mounts(request.user_id, session.expert_id),
            expert_id=session.expert_id,
            user_id=request.user_id,
            count_turn=False,
        )
    except Exception:
        logger.warning("[PAI] E2B sandbox setup failed", exc_info=True)
        return None


async def _load_history(user_id: str | None, session: ChatSession) -> _Loaded:
    """The stored history plus any rows written after it, or the rows alone.

    Runs before the turn writes any row of its own, so the watermark gap is
    exactly what other turns (or other engines) wrote since.
    """
    stored: LoadedHistory | None = None
    upload_safe = True
    if user_id and len(session.messages) > 1:
        upload_safe, stored = await download_history(user_id, session.session_id)
    if stored is None:
        rows = await extract_context_messages(
            None, session.messages, session_id=session.session_id
        )
        return _Loaded(
            history=chat_rows_to_messages(rows), held=[], upload_safe=upload_safe
        )
    gap = detect_gap(
        TranscriptDownload(content=b"", message_count=stored.watermark),
        session.messages,
    )
    if not gap:
        return _Loaded(
            history=stored.messages, held=stored.held, upload_safe=upload_safe
        )
    # Rows from elsewhere follow the held calls, so answer those in place.
    closed = close_pending_calls(
        stored.messages, {held.tool_call_id: held.output for held in stored.held}
    )
    return _Loaded(
        history=[*closed, *chat_rows_to_messages(gap)], held=[], upload_safe=upload_safe
    )


def _maybe_title(
    session: ChatSession, message: str | None, request: TurnRequest
) -> None:
    if not request.is_user_message or session.title:
        return
    user_rows = [m for m in session.messages if m.role == "user"]
    if len(user_rows) != 1:
        return
    first = user_rows[0].content or message or ""
    if first:
        task = asyncio.create_task(
            _update_title_async(request.session_id, first, request.user_id)
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)


async def _static_instructions(
    base_prompt: str,
    session: ChatSession,
    request: TurnRequest,
    graphiti: bool,
    experts: bool,
) -> str:
    return build_static_instructions(
        PromptInputs(
            base_system_prompt=base_prompt,
            graphiti_enabled=graphiti,
            experts_enabled=experts,
            expert_id=session.expert_id,
            source_platform=session.metadata.source_platform,
            autopilot_mode=await active_mode(request.user_id, session),
            builder_session_suffix=await build_builder_system_prompt_suffix(session),
            expert_session_suffix=await build_expert_identity_suffix(
                session.user_id,
                session.expert_id,
                organization_id=session.organization_id,
                team_id=session.team_id,
            ),
        )
    )


def _disabled(
    session: ChatSession, graphiti: bool, experts: bool
) -> tuple[list[ToolGroup], frozenset[str]]:
    groups: list[ToolGroup] = [] if graphiti else ["graphiti"]
    groups.extend(
        expert_tool_disabled_groups(
            experts_enabled=experts, expert_id=session.expert_id
        )
    )
    tools = (
        kickoff_turn_disabled_tools()
        if is_expert_kickoff_turn(session)
        else origin_disabled_tools(session.metadata.origin)
    )
    return groups, tools


def _set_context(
    request: TurnRequest,
    session: ChatSession,
    sandbox: Any,
    working_dir: str | None,
    groups: list[ToolGroup],
    disabled: frozenset[str],
) -> None:
    """The baseline's execution context, with ``run_capability`` bounded by
    the same hidden set that shaped the tool list."""
    set_execution_context(
        request.user_id,
        session,
        sandbox=sandbox,
        sdk_cwd=working_dir,
        permissions=request.permissions,
        envelope=request.envelope,
        hidden_tools=(
            tool_names_in_groups(groups)
            | disabled
            | denied_tool_names(request.permissions)
        ),
    )


async def _attachments(
    request: TurnRequest, working_dir: str | None
) -> tuple[str, list[dict]]:
    if not (request.file_ids and request.user_id and working_dir):
        return "", []
    return await _prepare_baseline_attachments(
        request.file_ids, request.user_id, request.session_id, working_dir
    )


def late_results_prompt(
    user_prompt: str | list[UserContent] | None, late: list[PendingMessage]
) -> str | list[UserContent] | None:
    """Add answered held calls that no pending tool call takes to the prompt,
    as the baseline adds them: after everything else the user sent."""
    if not late:
        return user_prompt
    extra: list[str] = [format_pending_as_user_message(pm)["content"] for pm in late]
    if user_prompt is None:
        if len(extra) == 1:
            return extra[0]
        only_late: list[UserContent] = [*extra]
        return only_late
    base: list[UserContent] = (
        [user_prompt] if isinstance(user_prompt, str) else list(user_prompt)
    )
    return [*base, *extra]
